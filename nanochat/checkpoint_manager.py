"""
Utilities for saving and loading model/optim/state checkpoints.
检查点管理: 模型保存/加载、断点续训、向后兼容补丁。

文件命名:
  model_005000.pt      — 模型参数 (torch.save)
  optim_005000_rank0.pt — 优化器状态 (每rank一个，分布式时各rank独立存储)
  meta_005000.json      — 元数据 (配置、步数、数据加载器状态、循环状态)
"""
import os
import re
import glob
import json
import logging
import torch
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    """仅rank 0打印日志(分布式训练时避免重复输出)"""
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints.
    向后兼容: 旧检查点缺少新增的配置键时自动补默认值"""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0(f"Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints.
    向后兼容: 旧模型权重缺少新增参数时自动创建默认值"""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)，残差缩放默认为1(=无缩放)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)，初始嵌入混合默认为0(=不混合)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0(f"Patching missing x0_lambdas in model data to 0.0")

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    """保存检查点: 模型参数、元数据、优化器状态。
    Save checkpoint: model params (rank 0 only), metadata JSON (rank 0 only), and optimizer shard (each rank).

    Args:
        checkpoint_dir: 检查点目录
        step: 当前训练步数
        model_data: 模型参数字典 (仅rank 0保存)
        optimizer_data: 优化器状态字典 (可为None; 分布式时每rank保存自己的分片)
        meta_data: 元数据字典 (配置、数据加载器状态等, 仅rank 0保存)
        rank: 当前进程rank
    """
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters / 保存模型参数
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json / 保存元数据为JSON
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    # 优化器状态在分布式训练中是分片的, 每个rank保存自己的分片
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    """加载检查点: 模型参数、元数据、可选的优化器状态分片。
    Load checkpoint: model params, metadata JSON, and optionally optimizer shard for a given rank.

    Args:
        checkpoint_dir: 检查点目录
        step: 要加载的步数
        device: 目标设备 (torch.device)
        load_optimizer: 是否加载优化器状态
        rank: 当前进程rank (加载优化器分片时需要)

    Returns:
        (model_data, optimizer_data, meta_data) 三元组, optimizer_data 可能为 None
    """
    # Load the model state / 加载模型参数
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested / 按需加载优化器状态
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata / 加载元数据
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    从检查点构建完整模型 (加载参数 + 分词器 + 元数据)。

    流程:
      ① 加载 model_{step}.pt + meta_{step}.json
      ② CPU/MPS 时 bf16→fp32 转换
      ③ 修复 torch.compile 的 _orig_mod. 前缀
      ④ 向后兼容补丁 (_patch_missing_*)
      ⑤ 在 meta device 上构建模型外壳 → to_empty 分配显存 → init_weights 初始化
      ⑥ load_state_dict 覆盖为保存的权重
      ⑦ 加载分词器 → 验证词表大小一致

    Returns: (model, tokenizer, meta_data)
      - model: 未编译、未包装DDP的原始模型
      - tokenizer: RustBPETokenizer 实例
      - meta_data: 训练时保存的元数据字典
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference (CPU不支持bf16)
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    # torch.compile 会在 state_dict 的 key 前加 "_orig_mod." 前缀，去掉以匹配模型
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)  # 补旧检查点缺失的配置键
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)  # 补旧检查点缺失的参数
    # meta device: 只定义形状/类型，不分配显存 → to_empty 再分配 → init_weights 初始化
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)  # 分配未初始化的显存
    model.init_weights()  # 初始化全部参数（会被 load_state_dict 覆盖）
    model.load_state_dict(model_data, strict=True, assign=True)  # 覆盖为保存的权重
    if phase == "eval": model.eval()
    else: model.train()
    tokenizer = get_tokenizer()
    # 词表一致性校验
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    """自动选择最大的模型: 先匹配 d<数字> 取最大depth，失败则取最近更新的目录。"""
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) 目录名符合 d<number> 格式: 按depth排序取最大
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            candidates.append((int(match.group(1)), model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)  # depth降序
        return candidates[0][1]
    # 2) 都不符合: 取最近修改时间的目录
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """在指定目录中查找最新的检查点步数。
    Find the highest step number in a checkpoint directory by scanning model_*.pt files."""
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    # 扫描目录中所有 model_*.pt 文件, 提取步数并返回最大值
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """从检查点目录加载模型, 自动推断 model_tag 和 step。
    Load a model from a checkpoints directory, auto-guessing model_tag (largest depth) and step (latest).

    Args:
        checkpoints_dir: 检查点根目录 (包含 d012, d024 等子目录)
        device: 目标设备
        phase: "train" 或 "eval"
        model_tag: 模型标签 (如 "d012"), 为None时自动选最大的depth
        step: 检查点步数, 为None时自动选最新的步数

    Returns:
        (model, tokenizer, meta_data) 三元组
    """
    if model_tag is None:
        # guess the model tag by defaulting to the largest model / 默认选择depth最大的模型
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step / 默认选择最新的步数
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model / 构建模型
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    """便捷函数: 根据来源标识加载模型 (自动解析为标准目录路径)。
    Convenience: load a model by source identifier. Maps "base"/"sft"/"rl" to their standard checkpoint directories.

    Args:
        source: 模型来源, 可选 "base" | "sft" | "rl"
        *args, **kwargs: 传递给 load_model_from_dir 的后续参数
    """
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
