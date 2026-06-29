"""
nanochat 高效推理引擎——围绕 token 序列设计的极速推理模块。
Engine for efficient inference of our models.

所有操作基于token序列:
  - 用户发送token序列→引擎返回下一个token
  - 引擎只处理纯token ID，完全不知道分词的存在
Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

核心效率手段:
  - KV Cache: 历史K/V缓存→每步只算新token→O(N²)降为O(N)
  - FA3原生布局(B,T,H,D): 无需transpose，零拷贝
  - 批量推理: batch=N个独立样本共享prompt→一份KV Cache
  - 工具调用状态机: 自动拦截python_start→计算→注入output→无需model参与
Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# 计算器工具辅助函数 — 安全执行Python表达式（数学/字符串操作）
# Calculator tool helpers — safely evaluate Python expressions (math/string ops)
@contextmanager
def timeout(duration, formula):
    """超时上下文管理器: 使用SIGALRM信号，超时后抛异常。
    Timeout context manager using SIGALRM signal. Raises exception on timeout."""
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    """在超时保护下安全求值表达式。关闭builtins避免注入攻击。
    Evaluate an expression under timeout. No builtins for safety."""
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # 忽略计算器错误——不影响生成流程 / it's ok to ignore wrong calculator usage
        # print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

def use_calculator(expr):
    """
    安全计算Python表达式——支持纯数学表达式和字数统计(.count())字符串操作。
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()

    安全措施: 纯数学→禁**操作符；字符串→白名单字符集 + 禁危险模式列表 + 仅允许.count()。
    Safety: pure math → no **; strings → whitelist chars + blacklist dangerous patterns + only .count().
    """
    # 去掉数字中的逗号(千位分隔符) / Remove commas from numbers
    expr = expr.replace(",", "")

    # 纯数学表达式检测 / Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 禁止幂操作符 / disallow power operator
            return None
        return eval_with_timeout(expr)

    # 字符串操作: 允许字母、数字、引号、括号、下划线、空格、点
    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 危险模式黑名单 / Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前只允许.count()方法（后续可扩展） / Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # 超时保护求值 / Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    为Flash Attention 3的flash_attn_with_kvcache API设计的KV缓存。
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    推理加速核心: 缓存历史K/V→每步只算新token的注意力→O(N²)→O(N)。
    核心性能: 缓存历史Key/Value→每步只算新token→复杂度从O(N²)降到O(N)。

    与FA2风格的差异/Key differences from FA2-style cache:
    - 张量布局(B,T,H,D)而非(B,H,T,D) ← FA3原生布局，无需转置
    - Tensors are (B, T, H, D) not (B, H, T, D)  ← FA3原生布局
    - FA3在flash_attn_with_kvcache中原地更新缓存
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - 位置通过cache_seqlens张量逐batch元素跟踪
    - Position tracked per batch element via cache_seqlens tensor
    - prev_embedding: 前一token的归一化嵌入，供GPT Smear机制使用
    - prev_embedding: previous token normed embedding for GPT Smear mechanism
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        """预分配KV缓存张量。
        参数/Args:
          batch_size: 批量大小（prefill=1, decode=N）
          num_heads: KV头数（GQA时少于Q头数）
          seq_len: 最大序列长度
          head_dim: 每头维度
          num_layers: Transformer层数
          device/dtype: 缓存所在设备和精度"""
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # 预分配缓存: (n_layers, B, T, H, D) FA3原生布局 / FA3 native layout
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # 每batch元素的当前序列长度（FA3要求int32） / Current seq len per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # 前一token的归一化嵌入供smear使用（由模型前向设置）
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """重置缓存为空状态。Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """获取当前缓存位置（假设所有batch元素同步）。Get current position (all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """返回指定层的(k_cache, v_cache)视图。Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """推进缓存位置num_tokens步（最后一层处理后调用）。
        Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        从另一个缓存复制KV到本缓存——用于batch=1 prefill→batch=N decode的场景。
        Copy cached KV from another cache into this one.
        典型流程: batch=1处理prompt→prefill将KV广播到batch=N份→并行采样N条序列。
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        # 复制KV缓存 / copy KV across layers and batch
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # 复制smear状态: batch=1扩展→num_samples / Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
# 采样函数: 从logits采样下一个token
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """从给定logits (B, vocab_size)采样单个token。返回(B, 1)。
    Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1).

    温度=0 → argmax贪婪；温度>0 → softmax采样；top_k>0 → 仅取概率最高的k个。
    temp=0 → argmax greedy; temp>0 → softmax sample; top_k>0 → restrict to top-k."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)  # 取top-k logits
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------
# 行状态: 跟踪每行(样本)在生成过程中的状态
class RowState:
    """逐行状态跟踪——在批量生成中维护每个样本的独立状态。
    Per-row state tracking during generation — maintains independent state for each sample.

    状态字段:
      current_tokens: 当前token序列
      forced_tokens: 待强制注入的token队列(deque) — 用于注入工具调用结果
      in_python_block: 是否在python代码块内 — 触发表达式缓存
      python_expr_tokens: python表达式token缓存 — 遇python_end时求值
      completed: 是否已完成生成 — 遇assistant_end/BOS时置True
    """
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # 当前token序列 / Current token sequence for this row
        self.forced_tokens = deque() # 待强制注入的token队列 / Queue of tokens to force inject
        self.in_python_block = False # 是否在python代码块内 / Whether we are inside a python block
        self.python_expr_tokens = [] # 当前python表达式token / Tokens of the current python expression
        self.completed = False # 是否已完成生成 / Whether this row has completed generation

class Engine:
    """高效批量推理引擎——封装KV缓存、工具调用状态机和流式生成逻辑。
    Efficient batched inference engine — encapsulates KV cache, tool-use state machine, and streaming generation.

    核心价值/Core value:
      - GPT.generate(): O(N²), 无KV缓存, batch=1—适合调试
      - Engine.generate(): O(N), KV缓存+批量+工具调用—适合生产
    """

    def __init__(self, model, tokenizer):
        """初始化引擎。model: GPT模型实例, tokenizer: 用于编码/解码特殊标记(工具调用用)。
        Initialize engine. model: GPT instance, tokenizer: for encoding/decoding special tokens (tool use)."""
        self.model = model
        self.tokenizer = tokenizer # 工具调用需要 / needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        批量自回归生成(流式)+工具调用状态机。核心推理入口。
        流程: ① Prefill(batch=1处理prompt→填充KV Cache)
             ② 复制KV Cache到batch=N份
             ③ Decode循环: 每步N个样本→sample_next_token→状态机→yield→下一步logits
             ④ 自动拦截<|python_start|>...<|python_end|>→计算→注入结果
        效率: KV Cache把O(N²)降到O(N), 256token序列~20×加速vs GPT.generate()
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = COMPUTE_DTYPE  # KV缓存精度 = 模型计算精度 (GTX1630→f32, H100→bf16)

        # 获取工具调用状态机所需的特殊标记ID / Get special token IDs for the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # 采样到此→结束 / if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # 采样到此→结束 / if sampled, ends row

        # ① batch=1 prefill: 用prompt完整序列一次性填充KV Cache
        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # 复制1份logits到num_samples份 / (num_samples, vocab_size)

        # ② 复制KV缓存到batch=N份，多样本并行解码
        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # 释放prefill缓存的内存 / no need to keep this memory around

        # ③ 初始化每个样本的行状态 / 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # ④ 主生成循环 / 4) Main generation loop
        num_generated = 0
        while True:
            # 停止条件1: 达到最大token数 / Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件2: 所有行都已完成 / Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # 为每行采样下一个token / Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # 处理每行: 选择token(采样/强制)→更新状态→可选工具调用
            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # 每行的下一个token ID / contains the next token id along each row
            token_masks = [] # mask: 1=采样得到的, 0=强制注入的 / 1 if sampled, 0 if forced
            for i, state in enumerate(row_states):
                # 选择该行的下一个token: 优先用forced_tokens队列
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # 是否有待强制注入的token? / are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # 强制=0, 采样=1 / mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # 更新行状态: 追加token / Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # 遇到<|assistant_end|>或BOS→标记行完成 / On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # 工具调用状态机 / Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            # 注入输出: <output_start> + 结果 + <output_end>
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # ⑤ yield当前列的token和mask给调用方 / Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # ⑥ 准备下一轮迭代的logits / Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        非流式批量生成: 收集全部token一次性返回（评估/RL用）。
        Non-streaming batch generation that just returns the final token sequences.

        返回: (results, masks)
          - results: list[list[int]], 每条样本的完整token序列(不含终止标记)
          - masks: list[list[int]], 对应mask(1=采样, 0=强制注入)
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True  # 终止标记不包含在结果中 / terminal tokens not in results
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # 所有行完成则停止 / Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    快速内联测试——验证朴素model.generate()与高效Engine.generate()输出一致。
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.

    测试流程: 用相同prompt→分别跑两种生成→对比结果→输出匹配状态和时间对比。
    Test: same prompt → both generators → compare results → print match status and timing.
    """
    import time
    # 初始化计算环境 / init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # 加载模型和分词器 / load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # 通用超参数: 最多64token, 温度0(贪婪) / common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # 设置起始prompt / set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # 用model.generate()生成参考序列(朴素O(N²) / naive O(N²))
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # 用Engine.generate()生成(高效KV缓存 / efficient KV cache)
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # 注意: fp32精度运行 / note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # 仅打印第一行 / only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # 逐位比较两种生成结果 / compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
