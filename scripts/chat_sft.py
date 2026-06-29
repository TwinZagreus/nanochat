"""
对模型进行监督微调（Supervised Fine-Tuning, SFT）。

Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

或使用 torchrun 进行分布式训练：

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from nanochat.common import preflight_compile_check
preflight_compile_check()
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI 命令行参数
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model / 对模型进行监督微调（SFT）")
# 日志记录
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging) / wandb 运行名称（'dummy' 禁用 wandb 日志）")
# 运行时
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect) / 设备类型（空=自动检测）")
# 模型加载
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from / 要加载的模型标签")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from / 要加载的模型步数")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes) / 从预训练检查点热启动优化器（0=否, 1=是）")
# 训练范围
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch) / 优化步数（-1 = 完整 epoch）")
# 批次大小（默认：继承自预训练检查点）
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain) / 最大上下文长度（默认：继承自预训练）")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain) / 每个设备的批次大小（默认：继承自预训练）")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain) / 总批次大小（以 token 计）（默认：继承自预训练）")
# 优化（默认：继承自预训练检查点）
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain) / 嵌入参数学习率（Adam）（默认：继承自预训练）")
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain) / 反嵌入参数学习率（Adam）（默认：继承自预训练）")
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain) / 矩阵参数学习率（Muon）（默认：继承自预训练）")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR / 初始学习率占基础学习率的比例")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup / 学习率预热占总迭代的比例")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown / 学习率衰减占总迭代的比例")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR / 最终学习率占初始学习率的比例")
# 编译
# Compilation
parser.add_argument("--no-compile", action="store_true", help="disable torch.compile (useful for Windows without MSVC compiler) / 禁用 torch.compile（适用于没有 MSVC 编译器的 Windows）")
# 评估
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable) / 每 N 步评估验证集 bpb（-1 = 禁用）")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on / 用于评估验证损失的 token 数量")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable) / 每 N 步评估 ChatCORE 指标（-1 = 禁用）")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE / ChatCORE 分类任务每题最多使用的样本数")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE / ChatCORE 生成任务每题最多使用的样本数")
# 数据混合
# Data mixture
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice) / 训练混合中 MMLU 的 epoch 数（教授多选题）")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use) / 训练混合中 GSM8K 的 epoch 数（教授数学和工具使用）")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# 计算初始化
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU 对 CPU/MPS 无意义 / MFU not meaningful for CPU/MPS

# wandb 日志初始化
# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Flash Attention 状态
# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient. / Flash Attention 3 不可用，使用 PyTorch SDPA 作为后备方案。训练效率会降低。")

# 加载模型和分词器
# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# 从预训练检查点继承训练超参数（None = 继承，显式值 = 覆盖）
# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
use_compile = not os.environ.get("TORCH_COMPILE_DISABLE")
if use_compile:
    model = torch.compile(model, dynamic=False)
    print0("Model compiled with torch.compile (dynamic=False) / 模型已用 torch.compile 编译（dynamic=False）")
else:
    print0("WARNING: torch.compile is disabled. Training will run in eager mode (slower). / torch.compile 已禁用。训练将以 eager 模式运行（更慢）。")
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 单 rank 每次迭代的 token 数 / tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有 rank 每次迭代的总 token 数 / total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps} / 总批次大小 {args.total_batch_size:,} => 梯度累积步数：{grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# 初始化优化器（组合 MuonAdamW：Muon 用于矩阵参数，AdamW 用于其余参数）
# 注意：预训练会在结束时将 weight_decay 衰减到零，因此 SFT 从零 weight_decay 继续
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# 可选：从预训练检查点热启动优化器（动量缓冲区等）
# 注意：load_state_dict 会用预训练值覆盖 param_group 元数据（学习率、betas 等）。
# 由于预训练的 warmdown 会将学习率降至 ~0，我们必须在加载后保存并恢复 SFT 的新学习率。
# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset) / 从预训练检查点加载了优化器状态（仅动量缓冲区，学习率已重置）")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse) / 优化器检查点未找到，使用全新优化器启动（效果略差）")

# fp16 训练的 GradScaler（bf16/fp32 不需要）
# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training / GradScaler 已启用（fp16 训练）")

# 将初始学习率覆盖为基础学习率的指定比例
# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT 数据混合与 DataLoader
# SFT data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [
    SmolTalk(split="train"), # 460K 行通用对话 / 460K rows of general conversations
    CustomJSON(filepath=identity_conversations_filepath), # 1000 行合成身份对话 / 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # 该数据集的第 2 个 epoch / 2 epochs of these
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 每个 epoch 100K 行 / 100K rows per epoch
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 每个 epoch 8K 行 / 8K rows per epoch
    SimpleSpelling(size=200000, split="train"), # 200K 行简单拼写（如拼写单词 'apple'） / 200K rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=80000, split="train"), # 80K 行拼写蜂（如 'strawberry' 中有几个 'r'） / 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]
train_dataset = TaskMixture(train_tasks)
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K 行测试集 / 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K 行测试集，仅用 5.2K 以匹配训练比例 / 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K 行测试集，仅用 420 以匹配训练比例 / 1.32K rows in test set, use only 420 to match the train ratios
]) # 总计：24K + 5.2K + 0.42K ~= 29.6K 行 / total: 24K + 5.2K + 0.42K ~= 29.6K rows
# 这里定义 DataLoader，它产生 inputs, targets 两个二维张量，形状为 (device_batch_size, max_seq_len)
# 一个主要问题是：我们无法提前知道最终的 num_iterations。因此创建
# 这两个全局变量，并在数据生成器内部更新它们。
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # 当到达训练数据集末尾时将其置为 True / we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # 在一个 epoch 过程中从 0 变为 1 / will go from 0 to 1 over the course of the epoch
current_epoch = 1 # 跟踪 epoch 用于日志记录 / track epoch for logging
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    SFT 的 BOS 对齐数据加载器，采用 bestfit-pad 打包策略。

    批次中的每一行以 BOS（对话开始标记）开头。
    对话使用最佳适配算法打包。当没有对话能放入时，
    该行会被填充（而非裁剪），确保不会丢弃任何 token。
    填充位置的目标值被掩码为 -1（交叉熵的 ignore_index）。

    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val' / split 必须是 'train' 或 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 为最后一个位置的 target 预留 / +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # 对话缓冲区：由 (token_ids, loss_mask) 元组组成的列表
    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # 每个 rank 处理不同的对话（拉取时错开） / Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # 独立于缓冲的消费计数 / Track actual consumption separately from buffering
    epoch = 1
    it = 0  # 迭代计数器 / iteration counter

    def refill_buffer():
        """当缓冲区数据不足时，从数据集中获取更多对话填充缓冲区。

        Refill the conversation buffer with more conversations from the dataset."""
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # 注意：last_step 现在基于消费计数而非抓取计数来触发
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []
        mask_rows = []
        row_lengths = []  # 跟踪每行的实际内容长度（不含填充）/ Track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # 确保缓冲区中有足够的对话
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # 找到能完全放入的最大对话
                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # 找到能放入的对话——完整使用它
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # 跟踪实际消费 / Track actual consumption
                else:
                    # 没有对话能放入——填充剩余空间，而非裁剪对话
                    # 这确保了永不丢弃任何 token
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # 用 BOS token 填充 / Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # 该行现已填满（通过填充）/ Row is now full (with padding)

            # 跟踪内容长度：未填充则为完整行容量，否则为填充前的长度
            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # 终止条件：如果指定了 num_iterations，则在达到后停止
        # Stopping condition to respect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # 更新进度跟踪（基于消费计数而非游标，以考虑缓冲的影响）
        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            # 当消费足够时触发 last_step（而非游标回绕时触发）
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if consumed >= dataset_size:
                last_step = True

        # 构建张量
        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # 应用来自 render_conversation 的损失掩码（mask=1 表示 assistant 的补全内容，
        # mask=0 表示用户提示、BOS、特殊 token、工具输出）。mask[1:] 与
        # targets 对齐（偏移 1）。未被掩码的位置置为 -1（ignore_index）。
        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # 掩码 targets 中的填充位置（设为 -1 = ignore_index）
        # 对于每一行，targets 中位置 >= (content_length - 1) 的部分需要被掩码
        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # 在一个 epoch 过程中从 0 变为 1 / will go from 0 to 1 over the course of the epoch

# 学习率调度器（线性预热、恒定、线性衰减）
# 形状与 base_train 相同，但使用 progress (0→1) 而非绝对步数，
# 因为 SFT 并不总能提前知道 num_iterations（由数据集驱动的停止策略）。
# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    """根据训练进度计算学习率乘数。预热从 0 线性增长到 1，衰减期线性降到 final_lr_frac。

    Calculate learning rate multiplier based on training progress. Warms up from 0 to 1, then decays linearly to final_lr_frac.
    """
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Muon 优化器的动量调度器
# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    """计算 Muon 优化器的动量值。前 300 步从 0.85 线性升温到 0.95。

    Calculate Muon optimizer momentum. Warms up from 0.85 to 0.95 over the first 300 steps.
    """
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 训练循环
# Training loop
x, y = next(train_loader) # 预取第一批数据 / prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # 训练损失的 EMA / EMA of training loss
ema_beta = 0.9 # EMA 衰减因子 / EMA decay factor
total_training_time = 0 # 训练总墙钟时间 / total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # 在分布式环境中跨所有 rank 同步 last_step，避免死锁
    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # 定期评估：验证集 bpb（所有 rank 参与）
    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 定期评估：ChatCORE 指标（所有 rank 参与）
    # 使用原始未编译模型，因为输入形状会变化
    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # 计算 ChatCORE 指标（平均中心化准确率，范围从 0=随机 到 1=完美）
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            """计算一组任务的平均中心化准确率。

            Compute the mean centered accuracy across a set of tasks."""

            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # 运行结束时保存检查点（所有 rank 参与，各自保存其优化器分片）
    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # 如 d12 / e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # 最后一步的损失 / loss at last step
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config, # 训练脚本的输入参数 / inputs to the training script
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单步训练
    # 计算梯度
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # 用于日志记录 / for logging
        loss = loss / grad_accum_steps # 每个 .backward() 是梯度累加 => 在此处归一化损失 / each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(train_loader) # GPU 忙于前向/反向传播时预取下一批数据 / prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # 进度只单调增加 / only increase progress monotonically
    # 执行优化器步进
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 状态更新
    # State
    step += 1

    # 日志记录
    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # 对训练损失做 EMA 平滑 / EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 对 EMA 做去偏处理 / debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # 仅计入前 10 步之后的时间 / only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

    # 垃圾回收器经常花费约 500ms 扫描循环引用。
    # 我们手动管理 GC 以避免训练过程中的这些暂停。
    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect() # 手动回收初始化阶段产生的大量垃圾 / manually collect a lot of garbage from setup
        gc.freeze() # 冻结所有当前存活对象，将它们排除出 GC 扫描范围 / freeze all currently surviving objects and exclude them from GC
        gc.disable() # 禁用 GC，除了以下情况： / disable GC entirely except:
    elif step % 5000 == 0: # 每 5000 步... / every 5000 steps...
        gc.collect() # 手动回收，仅为超长训练运行提供保障 / manually collect, just to be safe for very long runs

# 打印更多统计信息
# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# 记录到报告
# Log to report
from nanochat.report import get_report
get_report().log(section="SFT / 监督微调", data=[
    user_config, # CLI 参数 / CLI args
    { # 训练设置的统计信息 / stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # 训练结果的统计信息 / stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# 清理
# cleanup
wandb_run.finish() # 结束 wandb 运行 / wandb run finish
compute_cleanup()
