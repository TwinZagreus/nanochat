"""
nanochat通用工具集 / Common utilities for nanochat.
"""

import os
import sys
import re
import time
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# 计算精度(矩阵乘法/激活值)。主权重保持fp32以保证优化器精度。
# 线性层在前向传播中将权重转换为此精度,替代torch.amp.autocast。
# 通过NANOCHAT_DTYPE环境变量覆盖: "bfloat16", "float16", "float32"
# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    """
    自动检测计算精度(模块加载时运行一次，结果存 COMPUTE_DTYPE 全局变量)。
    优先级: NANOCHAT_DTYPE环境变量 > GPU SM版本检测 > 默认fp32(CPU/MPS)
    SM≥8.0(H100/A100)→bf16, SM<8.0(GTX1630/V100/T4)→fp32, 无CUDA→fp32
    """
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器,为日志消息添加颜色。 / Custom formatter that adds colors to log messages."""
    # ANSI颜色代码 / ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # 青色 / Cyan
        'INFO': '\033[32m',     # 绿色 / Green
        'WARNING': '\033[33m',  # 黄色 / Yellow
        'ERROR': '\033[31m',    # 红色 / Red
        'CRITICAL': '\033[35m', # 品红 / Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # 为级别名称添加颜色 / Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # 格式化消息 / Format the message
        message = super().format(record)
        # 为消息特定部分添加颜色 / Add color to specific parts of the message
        if levelname == 'INFO':
            # 高亮数字和百分比 / Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    """配置默认日志: INFO级别 + ANSI彩色控制台输出。 / Set up default logging: INFO level with ANSI-colored console output."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    """
    获取nanochat基础目录(默认 ~/.cache/nanochat, 可通过NANOCHAT_BASE_DIR环境变量覆盖)。
    Get the base directory for nanochat (default ~/.cache/nanochat, overridable via NANOCHAT_BASE_DIR env var).
    """
    # 将nanochat中间文件与其他缓存数据共存于~/.cache(默认) / co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    从URL下载文件到基础目录的本地路径。
    使用文件锁防止多rank并发下载。连接错误时最多重试5次(常见于不稳定的网络)。

    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    Retries up to 5 times on connection errors (common on unstable networks).
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path  # 文件已存在,无需下载 / File already exists, skip download

    with FileLock(lock_path):
        # 获取锁后重新检查,避免竞态 / Recheck after acquiring lock to avoid race
        if os.path.exists(file_path):
            return file_path

        # 带重试的下载(用于不稳定连接) / Download with retries for flaky connections
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"Downloading {url}..." + (f" (attempt {attempt}/{max_attempts})" if attempt > 1 else ""))  # 下载中 / Downloading
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                break  # 下载成功 / success
            except (OSError, Exception) as e:
                if attempt == max_attempts:
                    raise  # 所有重试均失败 / all attempts exhausted
                wait = 2 ** attempt  # 指数退避 / exponential backoff
                print(f"Download failed: {e}. Retrying in {wait}s...")  # 下载失败,等待重试 / Download failed, retrying
                time.sleep(wait)

        # 写入本地文件 / Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")  # 已下载到 / Downloaded to

        # 如果提供了后处理函数则执行 / Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    """
    仅在rank 0上打印(分布式训练时避免多GPU重复输出)。
    当遇到Windows GBK编码错误时自动降级为ASCII输出。

    Prints only on rank 0 (avoids duplicate output in distributed training).
    Falls back to ASCII output on Windows GBK encoding errors.
    """
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        try:
            print(s, **kwargs)
        except UnicodeEncodeError:
            # Windows GBK控制台回退: 编码为ASCII,替换无法渲染的字符 / Windows GBK console fallback: encode to ASCII, replacing unrenderable chars
            print(s.encode('ascii', errors='replace').decode('ascii'), **kwargs)

def print_banner():
    """打印nanochat ASCII艺术横幅(仅rank 0)。 / Print nanochat ASCII art banner (rank 0 only)."""
    # 酷炫的DOS Rebel字体ASCII横幅, 由 https://manytools.org/hacker-tools/ascii-banner/ 生成
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    try:
        print0(banner)
    except UnicodeEncodeError:
        # Windows GBK控制台无法渲染ASCII艺术,优雅跳过 / Windows GBK console can't render the ASCII art, skip gracefully
        print0("[nanochat]")

def is_ddp_requested() -> bool:
    """
    检查是否由torchrun启动(环境变量存在),用于判断是否需要初始化进程组。
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    检查torch.distributed是否可用且进程组已初始化。用于清理时避免销毁不存在的PG。
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    """
    获取分布式信息: (is_ddp, ddp_rank, ddp_local_rank, ddp_world_size)。
    如果非DDP模式,返回(False, 0, 0, 1)。

    Get distributed info tuple. Returns (False, 0, 0, 1) for non-DDP mode.
    """
    if is_ddp_requested():
        # 依赖torchrun的环境变量判断是否应初始化 / We rely on torchrun's env to decide if we SHOULD init.
        # (初始化本身在compute_init中完成) / (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    """
    自动检测设备类型: 优先CUDA → 其次MPS → 最后CPU。
    Autodetect device type: prefer CUDA if available → MPS → CPU fallback.
    """
    # 优先CUDA,否则MPS,最后回退CPU / prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")  # 自动检测设备类型 / Autodetected device type
    return device_type

def compute_init(device_type="cuda"): # 设备类型: cuda|cpu|mps / device type: cuda|cpu|mps
    """
    基础计算初始化(种子/精度/分布式),每次脚本启动都需要,所以提取为通用函数。
    Basic init (seed/precision/distributed) needed at every script start, so reified as common.
    """

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # 可复现性: 设置全局种子,但大部分代码使用显式rng对象。
    # 唯一可能使用全局rng的地方是nn.Module的模型权重初始化。
    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # 暂不追求完全可复现性,可能以后研究性能影响 / skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # 精度: CUDA上设置tf32矩阵乘法(速度更快,精度略低于fp32)
    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # 使用tf32代替fp32进行矩阵乘法,参见 / uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # 分布式设置: DDP可选,需要CUDA / Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # 使"cuda"默认指向此设备 / make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")  # 分布式世界大小 / Distributed world size

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """compute_init的配套函数,在脚本退出前清理分布式资源。 / Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """当不使用wandb时提供相同签名的占位符,避免代码中的条件分支。 / Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# 各种GPU的BF16峰值浮点运算次数(硬编码)
# 灵感来自torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# 和PR: https://github.com/karpathy/nanochat/pull/147
# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    """
    根据GPU设备名称返回BF16峰值FLOPS。未知GPU返回inf(使MFU显示为0%)。
    Returns BF16 peak FLOPS for the given GPU device name. Returns inf for unknown GPUs (MFU = 0%).
    """
    name = device_name.lower()

    # 表顺序很重要: 更具体的模式排前面。 / Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell架构 / NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper架构 / NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere数据中心 / NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada数据中心 / NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA加速器 / AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # 消费级RTX显卡 / Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - 基于计算单元动态计算 / dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # 未知GPU - 返回inf使MFU显示为0%,避免错误猜测 / Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

# -----------------------------------------------------------------------------
# torch.compile能力预检 / Pre-flight torch.compile capability check

def preflight_compile_check():
    """
    检查torch.compile在当前系统上是否可用,若不可用则设置TORCH_COMPILE_DISABLE=1。

    重要: 必须在导入nanochat.gpt(它会导入nanochat.optim)之前调用此函数。
    optim.py中的@torch.compile装饰器在导入时求值,因此环境变量必须在导入前设置。

    Check whether torch.compile can work on this system and set
    TORCH_COMPILE_DISABLE=1 if it cannot.

    IMPORTANT: Call this BEFORE importing nanochat.gpt (which imports
    nanochat.optim). The @torch.compile decorators in optim.py evaluate
    at import time, so the environment variable must be set before then.
    """
    # 1) 用户通过CLI标志显式请求禁用编译 / User explicitly requested no-compile via CLI flag
    if "--no-compile" in sys.argv:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        logger.info("torch.compile disabled via --no-compile flag")
        return

    # 2) 用户通过环境变量显式请求禁用编译 / User explicitly requested no-compile via environment variable
    if os.environ.get("NANOCHAT_NO_COMPILE", "0") == "1":
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        logger.info("torch.compile disabled via NANOCHAT_NO_COMPILE=1")
        return

    # 3) 自动检测: 尝试编译一个小函数来测试C++编译器是否可用 / Auto-detect: try a tiny torch.compile to see if the C++ compiler is available
    try:
        @torch.compile(dynamic=False)
        def _compile_smoke_test(x):
            return x.sin().cos()
        # 使用CPU张量来触发C++/OpenMP Inductor后端路径 / Use CPU tensor to exercise the C++/OpenMP Inductor backend path
        _compile_smoke_test(torch.randn(2, 2, device='cpu'))
        logger.info("torch.compile is available and working")  # torch.compile可用
    except Exception as e:
        logger.warning(
            f"torch.compile is not available on this system. "  # torch.compile在此系统上不可用
            f"Reason: {e}"  # 原因
        )
        logger.warning(
            "Training will continue without compilation (slower). "  # 训练将在无编译模式下继续(速度较慢)
            "Use --no-compile flag or set NANOCHAT_NO_COMPILE=1 to skip this check."  # 使用--no-compile或设置NANOCHAT_NO_COMPILE=1跳过此检查
        )
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
