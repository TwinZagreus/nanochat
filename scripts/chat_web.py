#!/usr/bin/env python3
"""
统一的 Web 聊天服务器 - 由单个 FastAPI 实例同时提供 UI 和 API 服务。
Unified web chat server - serves both UI and API from a single FastAPI instance.

使用数据并行将请求分发到多个 GPU。每个 GPU 加载完整的模型副本，
接收的请求被分发到可用的 worker 上。
Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

启动示例：
Launch examples:

- 单个可用 GPU（默认）
- single available GPU (default)
python -m scripts.chat_web

- 4 个 GPU
- 4 GPUs
python -m scripts.chat_web --num-gpus 4

要开始聊天，在浏览器中打开控制台打印的 URL。（如果在云服务器上，请确保使用公网 IP）
To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

接口：
Endpoints:
  GET  /           - 聊天界面 / Chat UI
  POST /chat/completions - 聊天 API（仅支持流式） / Chat API (streaming only)
  GET  /health     - 健康检查及 worker 池状态 / Health check with worker pool status
  GET  /stats      - Worker 池统计信息和 GPU 利用率 / Worker pool statistics and GPU utilization

防滥用限制：
Abuse Prevention:
  - 每个请求最多 500 条消息 / Maximum 500 messages per request
  - 每条消息最多 8000 字符 / Maximum 8000 characters per message
  - 会话总长度最多 32000 字符 / Maximum 32000 characters total conversation length
  - Temperature 限制在 0.0-2.0 / Temperature clamped to 0.0-2.0
  - Top-k 限制在 0-200（0 禁用 top-k 过滤，使用完整词表） / Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens 限制在 1-4096 / Max tokens clamped to 1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# 防滥用限制
# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 禁用 top-k 过滤，使用完整词表 / 0 disables top-k filtering, using full vocabulary
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web 服务器 / NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='使用的 GPU 数量（默认：1） / Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|rl / Source of the model: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='默认生成温度 / Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='默认 top-k 采样参数 / Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='默认生成最大 token 数 / Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签 / Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的步数 / Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='服务器运行端口 / Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='设备类型：cuda|cpu|mps，留空则自动检测 / Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器绑定的主机地址 / Host to bind the server to')
args = parser.parse_args()

# 配置对话流量的日志记录
# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    """在特定 GPU 上加载了模型的工作单元。 / A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """Worker 池，每个 worker 在不同 GPU 上持有一个模型副本。 / Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        """
        初始化 Worker 池。如未指定 GPU 数量，在 CUDA 环境下自动检测可用 GPU 数量，其他设备类型默认为 1。
        Initialize the worker pool. If num_gpus is not specified, auto-detect available GPUs for CUDA, default to 1 for other device types.
        """
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # 例如 cpu|mps / e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """在每个 GPU 上加载模型。 / Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "仅 CUDA 支持多个 worker/GPU。cpu|mps 不支持。 / Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # 例如 cpu|mps / e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """从池中获取一个可用的 worker。 / Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """将 worker 归还到池中。 / Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    """单条聊天消息模型，包含角色和内容。 / Single chat message model with role and content."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """聊天请求模型，包含消息列表和可选的生成参数。 / Chat request model with message list and optional generation parameters."""
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """验证聊天请求以防止滥用。 / Validate chat request to prevent abuse."""
    # 检查消息数量
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # 检查每条消息的长度和会话总长度
    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # 验证角色值
    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # 验证 temperature 参数
    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # 验证 top_k 参数
    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # 验证 max_tokens 参数
    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """在启动时将所有模型加载到各个 GPU 上。 / Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """提供聊天 UI 页面。 / Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # 替换 API_URL 为同源地址
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """提供 NanoChat Logo，用于网站图标和页头。 / Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """
    流式生成助手的回复。累积 token 并正确处理多字节 UTF-8 字符（如 emoji），
    在不完整的 UTF-8 序列上暂停输出，确保客户端收到的都是完整的字符。
    Generate assistant response with streaming. Accumulates tokens and properly handles multi-byte UTF-8 characters (like emojis),
    pausing output on incomplete UTF-8 sequences so clients only receive complete characters.
    """
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # 累积 token 以正确处理多字节 UTF-8 字符（如 emoji）
    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # 跟踪上次完整的 UTF-8 字符串（不含替换字符）
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        # 停止条件
        # Stopping criteria
        if token == assistant_end or token == bos:
            break

        # 将 token 追加到序列中
        # Append the token to sequence
        accumulated_tokens.append(token)
        # 解码所有累积的 token 以获得正确的 UTF-8 处理
        # Decode all accumulated tokens to get proper UTF-8 handling
        # 注意：decode 是一个相当高效的操作，基本上是查表和字符串拼接
        # Note that decode is a quite efficient operation, basically table lookup and string concat
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # 仅在文本不以替换字符结尾时才输出
        # Only emit text if it doesn't end with a replacement character
        # 这样可以确保不会发出不完整的 UTF-8 序列
        # This ensures we don't emit incomplete UTF-8 sequences
        if not current_text.endswith('�'):
            # 仅提取自上次干净解码以来的新文本
            # Extract only the new text since last clean decode
            new_text = current_text[len(last_clean_text):]
            if new_text:  # 仅在有新内容时才产出 / Only yield if there's new content
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """聊天补全接口（仅支持流式）- 使用 worker 池实现多 GPU 支持。 / Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # 基本验证，防止滥用
    # Basic validation to prevent abuse
    validate_chat_request(request)

    # 将传入的对话记录到控制台
    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # 从池中获取一个 worker（如果全部忙碌则等待）
    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # 构建对话 token 序列
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # 流式响应，完成后释放 worker
        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # 累积回复内容用于日志记录
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # 将助手的回复记录到控制台
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # 流式结束后将 worker 归还到池中
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # 确保即使出错也释放 worker
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """健康检查接口，返回服务器状态和 worker 池就绪情况。 / Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """获取 worker 池统计信息，包括总 worker 数、可用数和忙碌数。 / Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
