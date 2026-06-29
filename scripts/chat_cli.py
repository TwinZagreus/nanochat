"""
新的升级版聊天模式，因为自上一版本以来很多代码有了变化。
New and upgraded chat mode because a lot of the code has changed since the last one.

目前仅支持单 GPU 运行：
Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='与模型进行对话 / Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|rl / Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签 / Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的步数 / Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='向模型发送提示词，获取单次回复 / Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成温度 / Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k 采样参数 / Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='设备类型：cuda|cpu|mps，留空则自动检测 / Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# 初始化模型和分词器
# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# 聊天状态机所需的特殊 token
# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# 创建 Engine 以进行高效生成
# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # 从启动命令获取提示词
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # 从控制台交互式获取提示词
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # 处理特殊命令
    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # 将用户消息添加到对话中
    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # 启动助手生成
    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # 去掉 batch 维度 (num_samples=1) / pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # 必须确保 assistant_end token 是最后一个 token
    # 因此即使生成因达到 max_tokens 而提前结束，也要在末尾追加它
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # 在提示词模式下，只需要一次回复就退出
    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
