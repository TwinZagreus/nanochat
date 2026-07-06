"""
训练基座模型。从项目根目录运行：

Train model. From root directory of the project, run as:

python -m scripts.base_train

或分布式运行：

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

如果只在 CPU/Macbook 上运行，需要训练一个小得多的 LLM。示例：

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

# 预检：在导入使用 torch.compile 的模块之前，先检测 torch.compile 是否可用
# Pre-flight check: detect if torch.compile works before importing modules that use it
from nanochat.common import preflight_compile_check
preflight_compile_check()

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI 命令行参数
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model / 预训练基座模型")

# =============================================================================
# 日志记录（Logging）
# =============================================================================
parser.add_argument("--run", type=str, default="dummy",
    help="wandb 运行名称，'dummy' 跳过 wandb 日志（离线实验用），正式训练建议取有意义的名字如 'speedrun' / "
         "wandb run name; 'dummy' disables wandb logging (for offline experiments)")

# =============================================================================
# 运行时（Runtime）
# =============================================================================
parser.add_argument("--device-type", type=str, default="",
    help="设备类型：cuda|cpu|mps。空字符串=自动检测（CUDA>MPS>CPU 优先级） / "
         "Device type: cuda|cpu|mps. Empty = autodetect (CUDA > MPS > CPU)")

# =============================================================================
# FP8 训练（FP8 training）—— 仅 H100 (SM90+) 有效，可提速 ~1.5x，节省 ~30% 显存
# =============================================================================
parser.add_argument("--fp8", action="store_true",
    help="启用 FP8 混合精度训练，前向+反向的矩阵乘法用 FP8，其余保持 BF16。仅 H100+ 支持 / "
         "Enable FP8 mixed-precision training (matrix multiplies in FP8, rest in BF16). H100+ only")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"],
    help="FP8 缩放粒度：tensorwise=整个张量共享一个 scale（快，推荐）；rowwise=每行一个 scale（更精确但慢） / "
         "FP8 scaling granularity: tensorwise=one scale per tensor (fast, recommended); rowwise=one scale per row (more accurate, slower)")

# =============================================================================
# 编译（Compilation）
# =============================================================================
parser.add_argument("--no-compile", action="store_true",
    help="禁用 torch.compile。Windows 无 MSVC 编译器时必须加此参数，否则会报错 / "
         "Disable torch.compile. Required on Windows without MSVC compiler")

# =============================================================================
# 模型架构（Model architecture）—— depth 是唯一核心旋钮，其余自动推导
# =============================================================================
parser.add_argument("--depth", type=int, default=20,
    help="【核心参数】Transformer 层数。d4≈37M, d12≈354M, d20≈952M, d24≈1.6B (GPT-2级别)。"
         "宽度/注意力头数/学习率/训练步数等均由此自动推导 / "
         "【Core knob】Number of Transformer layers. d4≈37M, d12≈354M, d20≈952M, d24≈1.6B (GPT-2 scale). "
         "All other hyperparams are auto-derived from this")
parser.add_argument("--aspect-ratio", type=int, default=64,
    help="模型宽度比例：model_dim = depth × aspect_ratio（向上取整到 head_dim 倍数）。d20×64=1280 维 / "
         "Model width ratio: model_dim = depth × aspect_ratio (rounded up to head_dim multiple). d20×64=1280 dims")
parser.add_argument("--head-dim", type=int, default=128,
    help="每个注意力头的维度。注意力头数 = model_dim / head_dim。必须被 8 整除（FA3 要求） / "
         "Dimension per attention head. num_heads = model_dim / head_dim. Must be divisible by 8 (FA3 requirement)")
parser.add_argument("--max-seq-len", type=int, default=2048,
    help="最大上下文长度（token 数）。越长约消耗显存（Attention 是 O(n²)），VRAM 不足时可降至 1024 或 512 / "
         "Max context length in tokens. Longer = more VRAM (attention is O(n²)). Reduce to 1024/512 if OOM")
parser.add_argument("--window-pattern", type=str, default="SSSL",
    help="滑动窗口注意力模式，按层重复平铺。L=全上下文(完整 attention)，S=1/4 上下文(局部 attention)。"
         "最后一层始终为 L。例如 'SSSL' 表示每 4 层中有 1 层用全上下文。非 H100 GPU 建议用 'L'（SDPA 不支持窗口注意力） / "
         "Sliding window pattern tiled across layers. L=full context, S=1/4 context. "
         "Last layer always L. Non-H100 GPU: use 'L' (SDPA doesn't support sliding window)")

# =============================================================================
# 训练范围（Training horizon）—— 三个参数按优先级互斥：num_iterations > target_flops > target_param_data_ratio
# =============================================================================
parser.add_argument("--num-iterations", type=int, default=-1,
    help="显式指定训练步数，-1=不使用此参数。例如 --num-iterations=100 只跑 100 步（快速验证用） / "
         "Explicit number of training steps. -1 = disabled. E.g. --num-iterations=100 for quick smoke test")
parser.add_argument("--target-flops", type=float, default=-1.0,
    help="根据目标总算力（FLOPs）自动计算训练步数，-1=不使用。少用，通常用 token/params 比更方便 / "
         "Auto-compute steps to reach target total FLOPs. -1 = disabled. Rarely used")
parser.add_argument("--target-param-data-ratio", type=float, default=12,
    help="【默认】Token/参数比，自动计算训练步数。12=计算最优(Chinchilla≈20), 8=略欠训练(更快,GPT-2速通用)。-1=禁用 / "
         "【Default】Target token-to-param ratio. 12=compute-optimal (Chinchilla≈20), "
         "8=slightly undertrained (faster, used in speedrun). -1=disabled")

# =============================================================================
# 优化（Optimization）
# =============================================================================
parser.add_argument("--device-batch-size", type=int, default=32,
    help="每 GPU 每步处理的序列数。OOM 时优先降低此值：32→16→8→4→2→1。降低不会影响训练效果（梯度累积自动补偿） / "
         "Sequences per GPU per step. Reduce first if OOM: 32→16→8→4→2→1. "
         "Lower values don't hurt training (gradient accumulation compensates)")
parser.add_argument("--total-batch-size", type=int, default=-1,
    help="全局批次大小（以 token 计），-1=根据缩放定律自动计算最优值。手动设置如 524288。"
         "全局批次大小 ÷ (device_batch_size × max_seq_len × world_size) = 梯度累积步数 / "
         "Total batch size in tokens. -1 = auto-compute optimal via scaling laws. "
         "E.g. 524288. Accumulation steps = total / (device × seq_len × world_size)")
parser.add_argument("--embedding-lr", type=float, default=0.3,
    help="词嵌入层 (nn.Embedding) 的学习率（用 AdamW 优化）。嵌入层对 LR 较敏感，需要较大值 / "
         "Learning rate for word embedding layer (AdamW). Embeddings need relatively high LR")
parser.add_argument("--unembedding-lr", type=float, default=0.008,
    help="LM Head（输出投影层）的学习率（用 AdamW 优化）。与 embedding 不共享权重 (untied) / "
         "Learning rate for LM head / unembedding layer (AdamW). Not tied with embedding weights")
parser.add_argument("--weight-decay", type=float, default=0.28,
    help="Muon 优化器的权重衰减系数。仅在梯度方向与参数方向一致时才施加（cautious weight decay） / "
         "Weight decay coefficient for Muon optimizer. Applied cautiously: only when gradient aligns with parameter direction")
parser.add_argument("--matrix-lr", type=float, default=0.02,
    help="矩阵参数（所有 Linear 层权重）的基准学习率（用 Muon 优化）。实际 lr 会按缩放定律调整 / "
         "Base learning rate for matrix parameters (Muon optimizer). Actual LR is scaled by model size")
parser.add_argument("--scalar-lr", type=float, default=0.5,
    help="标量参数的学习率，如 resid_lambdas（残差缩放系数）、x0_lambdas（初始嵌入混合系数）等 / "
         "Learning rate for scalar parameters: resid_lambdas, x0_lambdas, etc.")
parser.add_argument("--warmup-steps", type=int, default=40,
    help="学习率线性预热步数。从 0 线性升至目标 LR，避免训练初期梯度不稳定 / "
         "Number of linear LR warmup steps. LR ramps from 0 to target, stabilizes early training")
parser.add_argument("--warmdown-ratio", type=float, default=0.65,
    help="学习率衰减开始的位置（占总步数的比例）。0.65 表示前 65% 常数 LR，后 35% 线性衰减 / "
         "Fraction of total steps where LR decay begins. 0.65 = 65% constant LR, then linear decay over last 35%")
parser.add_argument("--final-lr-frac", type=float, default=0.05,
    help="最终学习率 = 初始 LR × 此值。0.05 表示衰减到初始 LR 的 5%，实现充分收敛 / "
         "Final LR = initial LR × this fraction. 0.05 means decay to 5% of initial LR for good convergence")
parser.add_argument("--resume-from-step", type=int, default=-1,
    help="从指定步数恢复训练，-1=从头训练。会加载模型参数+优化器状态+数据加载器位置，无缝接续 / "
         "Resume training from this step. -1 = train from scratch. "
         "Loads model+optimizer+dataloader state for seamless continuation")

# =============================================================================
# 评估（Evaluation）—— 训练期间定期执行，不影响训练本身
# =============================================================================
parser.add_argument("--eval-every", type=int, default=250,
    help="每 N 步在验证集上评估一次 BPB（bits per byte）。-1=跳过。频繁评估花时间，250 是合理平衡 / "
         "Evaluate validation BPB every N steps. -1 = skip. 250 is a reasonable balance")
parser.add_argument("--eval-tokens", type=int, default=80*524288,
    help="验证集评估时使用的 token 总数（默认约 42M）。越多越精确但越慢 / "
         "Total tokens for validation BPB evaluation. More = more accurate but slower")
parser.add_argument("--core-metric-every", type=int, default=2000,
    help="每 N 步评估 CORE 指标（20+ 下游任务的综合得分）。很耗时，间隔设大一些。2000 是合理默认。-1=跳过 / "
         "Evaluate CORE metric every N steps (20+ downstream tasks). Expensive, so use larger intervals. -1 = skip")
parser.add_argument("--core-metric-max-per-task", type=int, default=500,
    help="CORE 评估时每个任务最多使用的样本数。减少可加速评估，500 是高效默认 / "
         "Max examples per task for CORE evaluation. Reduce to speed up. 500 is efficient default")
parser.add_argument("--sample-every", type=int, default=2000,
    help="每 N 步用当前模型生成文本样本（看训练进度）。-1=跳过 / "
         "Generate text samples every N steps to inspect training progress. -1 = skip")
parser.add_argument("--save-every", type=int, default=-1,
    help="每 N 步保存一次检查点（模型+优化器+元数据），-1=仅在训练结束时保存。检查点文件较大(数GB) / "
         "Save checkpoint every N steps. -1 = only save at end. Checkpoint files are large (several GB)")

# =============================================================================
# 输出（Output）
# =============================================================================
parser.add_argument("--model-tag", type=str, default=None,
    help="自定义模型标签，用于检查点目录名。默认 None → 自动取 'd{depth}'，如 d20 → base_checkpoints/d20/ / "
         "Custom model tag for checkpoint dir. Default None → auto 'd{depth}', e.g. d20 → base_checkpoints/d20/")

args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# 计算初始化与 wandb 日志
# Compute init and wandb logging

# ① 设备检测：命令行指定优先，否则自动检测（CUDA > MPS > CPU）
# Device detection: CLI arg takes priority, otherwise autodetect (CUDA > MPS > CPU)
device_type = autodetect_device_type() if args.device_type == "" else args.device_type

# ② 初始化分布式环境，返回：
#    ddp           - bool, 是否多卡分布式（world_size > 1）
#    ddp_rank      - int, 当前进程编号，0=主进程（负责日志/检查点/wandb），其他=worker
#    ddp_local_rank - int, 当前节点上的本地 GPU 编号
#    ddp_world_size - int, 总 GPU 数量
#    device         - torch.device, 当前进程绑定的计算设备，如 cuda:0 / cpu
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# ③ 只有 rank=0 的进程负责日志、保存检查点、wandb 上报，避免多卡重复操作
master_process = ddp_rank == 0

# ④ CUDA 同步函数：用于精确计时（确保 GPU 操作完成后再计时）。CPU/MPS 不需要同步，用空函数代替
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# ⑤ 查询 GPU 显存峰值（字节数），用于日志记录。CPU/MPS 始终返回 0
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# ⑥ 获取 GPU 峰值算力（TFLOPS），用于计算 MFU（Model FLOPs Utilization，模型算力利用率）
# MFU = 实际 FLOPs / 峰值 FLOPs，衡量 GPU 利用效率。如 H100 BF16 ≈ 989 TFLOPS
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)          # 如 "NVIDIA H100 80GB HBM3"
    gpu_peak_flops = get_peak_flops(gpu_device_name)         # 如 9.89e14 (989 TFLOPS)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # CPU/MPS 上 MFU 无意义，设为无穷大跳过计算

# ⑦ 打印当前计算精度及选择原因
# 如: "COMPUTE_DTYPE: torch.bfloat16 (CUDA SM90+ GPU detected)"
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention 状态
# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# 分词器用于评估，同时我们也需要词汇表大小来初始化模型
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# 初始化模型 — 三步：meta 构建 → 分配内存 → 初始化权重
# Initialize the Model — three steps: meta build → allocate memory → init weights
#
# 以冒烟测试 2g (d4, head_dim=128 默认, aspect_ratio=64) 为例走一遍数据流：
#   depth=4, base_dim=4×64=256, 向上取整到128倍数→ model_dim=256
#   num_heads = 256/128 = 2
#   GPTConfig: n_layer=4, n_embd=256, n_head=2, vocab=32768, seq_len=512
#
# 模型结构（每一层）：
#   Input token ids [B, T] → Embedding [B, T, 256]
#   → 4× Block（每个 Block = CausalSelfAttention + MLP）
#       Attention: Q: [256→256], K: [256→256], V: [256→256], 输出投影 [256→256]
#       MLP:       两层 Linear，中间维度 256×4=1024，激活 ReLU²
#   → RMSNorm → Linear(256→32768) LM Head → softcap tanh → loss
#
# 每层矩阵参数形状（d4, model_dim=256, head_dim=128）：
#   Attention:
#     c_attn   [256, 768]    QKV 三合一投影 (256×3=768)
#     c_proj   [256, 256]    注意力输出投影
#   MLP:
#     c_fc     [1024, 256]   升维 (256→1024)
#     c_proj   [256, 1024]   降维 (1024→256)
#   共 4 个矩阵 × 4 层 = 16 个 Linear 层
#   额外: Embedding [32768, 256], LM Head [32768, 256] (untied, 不共享)
#   总参数量 ≈ 3.4M（不含标量参数如 resid_lambdas 等）

def build_model_meta(depth):
    """在 meta 设备上为给定深度构建模型（仅形状/数据类型，无实际数据）。

    Build a model on meta device for a given depth (shapes/dtypes only, no data).

    meta 设备是 PyTorch 的"幽灵设备"——tensor 只有 shape/dtype，不消耗任何内存。
    作用：提前知道模型结构（参数量、每层 shape），用于计算训练步数和优化器配置。
    """
    # ① 计算模型维度：base_dim = depth × aspect_ratio (默认 4×64=256)
    #    向上取整到 head_dim (默认 128) 的最近整数倍，保证整除
    #    d4, head_dim=128: 256 已是 128 的倍数 → model_dim=256, num_heads=256/128=2
    #    d6, head_dim=128: 6×64=384 → 向上取整 → 384, num_heads=384/128=3
    #    d4, head_dim=64:  256 已是 64 的倍数  → model_dim=256, num_heads=256/64=4
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim

    # ② 组装 GPTConfig → 在 meta 设备上创建 GPT 对象
    #    此时所有 Linear/Embedding 层内部 tensor 都指向 meta 设备，不占显存
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# 三步构建模型：
#   Step 1: meta 设备构建（只知形状，不占内存）→ 拿到结构和参数量
#   Step 2: to_empty 分配真实显存（数据是垃圾值）
#   Step 3: init_weights 初始化所有参数（正态分布/零）
model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

# Step 2: 从 meta 设备 → 真实设备（CPU/GPU）
# to_empty() = 在目标设备上分配未初始化内存（数据是垃圾），比 torch.zeros 快（省去清零）
model.to_empty(device=device)

# Step 3: 初始化所有参数
# - Linear 权重: 正态分布 N(0, 1/sqrt(fan_in))，截断到 [-3σ, 3σ]
# - Embedding: 正态分布 N(0, 1/sqrt(embed_dim))
# - 标量参数 (resid_lambdas, x0_lambdas 等): 全 0 或全 1
# - 偏置: 无（nanochat 所有 Linear 层无 bias）
model.init_weights()

# 如果从检查点恢复训练，用检查点的参数覆盖模型参数
# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # 如 d12 / e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # 复制后释放内存 / free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 训练初始化和管理（必须在 torch.compile 之前完成）
# FP8 training initialization and management (this has to be done before torch.compile)

# 如果设置了 --fp8，将 Linear 层转换为 Float8Linear
# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag / FP8 训练需要 CUDA，忽略 --fp8 标志")
    else:
        # 我们自定义的 fp8 比 torchao 更简单，为完全兼容的 API 而编写
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # 过滤器：维度必须能被 16 整除（FP8 硬件要求），且足够大
        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small) / FP8 训练已启用（{args.fp8_recipe} 缩放）- 已转换 {num_fp8}/{num_linear} 个线性层，跳过 {num_skipped} 个（太小）")

# 上下文管理器：临时禁用 FP8，使模型评估保持在 BF16 精度
# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """临时将 Float8Linear 模块替换为 nn.Linear 以进行 BF16 评估。

    CastConfig 是一个冻结的数据类，无法修改 scaling_type。
    因此我们完全替换 Float8Linear 模块，并在之后恢复它们。

    Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # 找到所有 Float8Linear 模块及其位置
    # Find all Float8Linear modules and their locations
    fp8_locations = []  # 列表元素为 (父模块, 属性名, fp8模块) / list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # 没有 FP8 模块，无需操作 / No FP8 modules, nothing to do
        return

    # 将 Float8Linear 替换为 Linear（我们自定义的类，会将权重转换为匹配输入的数据类型）
    # 使用 device="meta" 避免显存飙升——之后会替换权重张量
    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # 使用 meta 设备避免不必要的显存分配 / Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # 共享而非复制 / share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # 恢复 Float8Linear 模块
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# 编译模型
# Compile the model

orig_model = model # 原始未编译模型，用于保存原始 state_dict 和推理/评估（因为形状可能会变化） / original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
use_compile = not os.environ.get("TORCH_COMPILE_DISABLE")
if use_compile:
    model = torch.compile(model, dynamic=False) # 模型的输入形状不会改变，所以 dynamic=False 是安全的 / the inputs to model will never change shape so dynamic=False is safe
    print0("Model compiled with torch.compile (dynamic=False) / 模型已用 torch.compile 编译（dynamic=False）")
else:
    print0("WARNING: torch.compile is disabled. Training will run in eager mode (slower). / torch.compile 已禁用。训练将以 eager 模式运行（更慢）。")
    # 模型保持为 orig_model（同一对象），所有评估路径正常工作
    # model stays as orig_model (same object), all eval paths work correctly

# -----------------------------------------------------------------------------
# 使用缩放定律和 muP 外推来确定最优训练范围、批次大小、学习率、权重衰减。
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# 获取模型的参数计数
# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) 使用缩放定律确定最优训练 token 数量
# 计算最优模型满足 --target-param-data-ratio 的 Token:Params 比率（通过缩放定律分析实验得出）。
# 模型已经初始化，Params 已知。最优 Token 数量 = target-param-data-ratio * Params
# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    """获取模型的缩放参数量：transformer 矩阵 + lm_head，这能产生最干净的缩放定律曲线。

    Get the number of scaling parameters: transformer matrices + lm_head, which gives the cleanest scaling laws.
    """
    # 关于使用哪些参数：transformer 矩阵 + lm_head 能产生最干净的缩放定律（参见 dev/LOG.md 2026年1月27日）
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # 当前模型的最优 token 数 / optimal tokens for the model we are about to train

# 参考模型是 d12，大部分超参数都在此调优，然后通过 muP 方式迁移到更深的模型
# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # 在 meta 设备上创建 d12 参考模型 / creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # d12 的计算最优训练 token 数（实验测得） / compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # d12 的最优批次大小 ~= 524,288 tokens（实验测得） / optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) 知道了 token 范围，可以计算最优批次大小
# 我们遵循 Power Lines 论文 (Bopt ∝ D^0.383)，参考：https://arxiv.org/abs/2505.13738
# 最优批次大小约按 D^0.383 增长，例如从 d12 翻倍到 d24，B 应增长 2^0.383 ≈ 1.3x。
# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # 用户可以覆盖 / user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # 取最近的 2 的幂以提高效率 / clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens / 自动计算的最优批次大小：{total_batch_size:,} tokens")

# 3) 知道了批次大小，可以计算学习率修正量（更大的批次允许更高的学习率）
# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref / B 相对 B_ref 的比率
if batch_ratio != 1.0:
    # SGD: 标准的线性缩放（nanochat 不使用）
    # AdamW: 标准的是 sqrt 缩放：η ∝ √(B/B_ref)
    # Muon: 我们对 Muon 采用与 AdamW 相同的缩放：η ∝ √(B/B_ref)（未经仔细研究，这是假设！）
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref) / 学习率正比于批次的平方根
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,}) / 将学习率缩放 {batch_lr_scale:.4f}，配合批次大小 {total_batch_size:,}（参考值：{B_REF:,}）")

# 4) 知道了批次大小和 token 范围，可以计算适当的权重衰减缩放
# 采用 T_epoch 框架，参考：https://arxiv.org/abs/2405.13698
# 论文核心思想：T_epoch = B/(η·λ·D) 应保持不变。
# 上面我们使用了学习率缩放 η ∝ √(B/B_ref)。
# 经过约 10 行数学推导：要保持 T_epoch 不变，需要：λ = λ_ref · √(B/B_ref) · (D_ref/D)
# 注意：这些论文研究的是 AdamW，不是 Muon。我们直接照搬 AdamW 的缩放理论，希望它对 Muon 也 ~有效。
# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth} / 将权重衰减从 {args.weight_decay:.6f} 缩放至 {weight_decay_scaled:.6f}，深度 {args.depth}")

# -----------------------------------------------------------------------------
# 初始化优化器（组合 MuonAdamW：Muon 用于矩阵参数，AdamW 用于其余参数）
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# fp16 训练的 GradScaler（bf16/fp32 不需要——bf16 与 fp32 有相同的指数范围）
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training / GradScaler 已启用（fp16 训练）")

# -----------------------------------------------------------------------------
# 初始化训练/验证数据加载器
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# 计算训练迭代次数并设置各种调度器
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations：可以是显式指定、根据目标 FLOPs 计算、或根据目标数据:参数比计算（按优先级）
# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# 学习率调度器（线性预热、恒定、线性衰减）
# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    """计算给定迭代步的学习率乘数。预热阶段从 0 线性增长到 1，训练末期线性衰减到 final_lr_frac。

    Calculate learning rate multiplier for a given iteration step. Linearly warms up from 0 to 1, then linearly decays to final_lr_frac at the end.
    """
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Muon 优化器的动量调度器（预热到 0.97，在 LR 衰减期间降至 0.90）
# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    """计算 Muon 优化器在当前迭代步的动量值。前 400 步从 0.85 线性升温到 0.97，LR 衰减期间降温到 0.90。

    Calculate Muon optimizer momentum at the given iteration. Warms up from 0.85 to 0.97 over the first 400 steps, then down to 0.90 during LR warmdown.
    """
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Muon 优化器的权重衰减调度器（在训练过程中余弦衰减到零）
# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    """计算当前迭代步的权重衰减值，在整个训练过程中从初始值余弦衰减到零。

    Calculate weight decay value at the given iteration, decaying from the initial value to zero via cosine over the training.
    """
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# 训练循环
# Training loop

# 循环状态（训练循环中更新的变量）
# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # 如果 eval_every > 0 则会被设置 / will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # 训练损失的 EMA / EMA of training loss
    total_training_time = 0 # 训练总墙钟时间 / total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# 计算达到所需总批次大小需要的梯度累积微步数
# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 单 rank 每次迭代的 token 数 / tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# 开始训练！
# Go!
while True:
    last_step = step == num_iterations # 循环运行 num_iterations+1 次，以便在结束时可以评估/保存 / loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 定期评估：验证集 bpb（所有 rank 参与）
    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 定期评估：CORE 指标（所有 rank 参与）
    # 使用原始未编译模型，因为输入形状会变化
    # 禁用 FP8 以使用 BF16 进行评估，获得更一致/准确的结果
    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # 定期采样：从模型生成文本（仅主进程）
    # 使用原始未编译模型，因为输入形状会变化
    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # 使用 orig_model 避免重新编译 / use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 保存检查点：运行结束时，或每 save_every 步（但不在第一步或恢复步骤处保存）
    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # 最后一步的损失 / loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # 训练脚本的输入参数 / inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # 所有循环状态（除 step 外），用于恢复训练 / all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # 终止条件（TODO: 可能还需添加损失爆炸等条件）
    # termination conditions (TODO: possibly also add loss explosions etc.)
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
        x, y, dataloader_state_dict = next(train_loader) # GPU 忙于前向/反向传播时预取下一批数据 / prefetch the next batch while the GPU is busy with forward/backward
    # 执行优化器步进
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # 在分布式训练中，所有 rank 必须就是否跳过该步达成一致。
        # 每个 rank 可能独立遇到 inf/nan 梯度，因此我们对 found_inf 标志进行 all-reduce
        # 取最大值（MAX = 如果有任何 rank 发现 inf，所有 rank 都跳过该步）。
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() 是 CPU-GPU 同步点 / .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 日志记录（仅 CPU 操作）
    # logging (CPU action only)
    ema_beta = 0.9 # EMA 衰减因子，用于平滑日志显示 / EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # 对训练损失做 EMA 平滑 / EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 对 EMA 做去偏处理 / debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # 仅计入前 10 步之后的时间 / only count the time after the first 10 steps
    # 根据每步平均时间计算 ETA（排除前 10 步）
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # 状态更新
    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # 垃圾回收器有时过于活跃，经常花费约 500ms 扫描循环引用，
    # 但最终只清理极少量小对象。因此我们手动管理 GC 以帮助优化。
    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # 手动回收初始化阶段产生的大量垃圾 / manually collect a lot of garbage from setup
        gc.freeze() # 立即冻结所有当前存活对象，将它们排除出 GC 扫描范围 / immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # 核武器级别的干预：完全禁用 GC，除了以下情况： / nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # 每 5000 步... / every 5000 steps...
        gc.collect() # 手动回收，仅为超长训练运行提供保障 / manually collect, just to be safe for very, very long runs

# 打印更多统计信息
# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# 记录到报告
# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training / 基座模型训练", data=[
    user_config, # CLI 参数 / CLI args
    { # 训练设置的统计信息 / stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # 训练结果的统计信息 / stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# 清理
# cleanup
wandb_run.finish() # 结束 wandb 运行 / wandb run finish
compute_cleanup()
