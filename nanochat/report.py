"""
生成训练报告卡片的工具函数。代码比平时更杂乱，后续会修复。
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

def run_command(cmd):
    """运行 shell 命令并返回输出，如果失败则返回 None。
    Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # 如果得到了输出就返回 stdout（即使 xargs 中某些文件失败了）
        # Return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except:
        return None

def get_git_info():
    """获取当前 git 提交、分支和脏状态。
    Get current git commit, branch, and dirty status."""
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # 检查仓库是否为脏状态（有未提交的更改）
    # Check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # 获取提交消息
    # Get commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # 第一行，截断 / First line, truncated

    return info

def get_gpu_info():
    """获取 GPU 信息。
    Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # 获取 CUDA 版本
    # Get CUDA version
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """获取系统信息。
    Get system information."""
    info = {}

    # 基本系统信息
    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU 和内存
    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # 用户和环境
    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """根据 GPU 类型和运行时间估算训练成本。
    Estimate training cost based on GPU type and runtime."""

    # 粗略定价，来自 Lambda Cloud
    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # 尝试从名称中识别 GPU 类型
    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # 默认估算 / Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """生成训练报告的头部。
    Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # 膨胀指标：仅统计 git 跟踪的源文件中的行数和字符数
    # bloat metrics: count lines/chars in git-tracked source files only
    extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
    git_patterns = ' '.join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or '').split('\n') if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            total_line = wc_output.strip().split('\n')[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                num_lines = int(parts[0])
                num_chars = int(parts[1])
    num_tokens = num_chars // 4  # 假设每个 token 大约 4 个字符 / assume approximately 4 chars per token

    # 通过 uv.lock 统计依赖项
    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r', encoding='utf-8') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """将文本字符串转换为 slug 格式。
    Slugify a text string."""
    # 替换路径分隔符和其他不安全的文件名字符
    for ch in "/\\:*?\"<>|":
        text = text.replace(ch, "-")
    return text.lower().replace(" ", "-")

# 预期存在的文件及其顺序
# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# 我们当前关注的指标
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """从报告部分提取指定键的值。
    simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys] # 便捷处理：允许传入单个键 / convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """从内容中提取具有给定前缀的时间戳。
    Extract timestamp from content with given prefix."""
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """维护一组日志，生成最终的 markdown 报告。
    Maintains a bunch of logs, generates a final markdown report."""

    def __init__(self, report_dir):
        """初始化报告对象，创建报告目录。"""
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """将一段数据记录到报告中。
        Log a section of data to the report."""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # 跳过虚值，如 None 或空字典等
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # 直接写入字符串
                    # directly write the string
                    f.write(item)
                else:
                    # 渲染字典
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """生成最终报告。
        Generate the final report."""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {} # 最终关键指标，将在末尾添加为表格 / the most important final metrics we'll add as table at the end
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # 首先写入头部
            # write the header first
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # 捕获膨胀数据供后续摘要使用（Bloat 标题之后直到 \n\n 的内容）
                    # capture bloat data for summary later (the stuff after Bloat header and until \n\n)
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None # 将导致不写入总挂钟时间 / will cause us to not write the total wall clock time
                bloat_data = "[bloat data missing]"
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat reset`?")
            # 处理所有单独的章节
            # process all the individual sections
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # 从此章节提取时间戳（最后一个章节的时间戳将作为 end_time 保留）
                # Extract timestamp from this section (the last section's timestamp will "stick" as end_time)
                if "rl" not in file_name:
                    # Skip RL sections for end_time calculation because RL is experimental
                    # 跳过 RL 章节的时间戳，因为 RL 是实验性的
                    end_time = extract_timestamp(section, "timestamp:")
                # 从各章节提取最重要的指标
                # extract the most important metrics from the sections
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K") # RL only evals GSM8K / RL 仅评估 GSM8K
                # 追加此章节到报告中
                # append this section of the report
                out_file.write(section)
                out_file.write("\n")
            # 添加最终指标表格
            # add the final metrics table
            out_file.write("## Summary\n\n")
            # 从头部复制膨胀指标
            # Copy over the bloat metrics from the header
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # 收集所有唯一的指标名称
            # Collect all unique metric names
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # 自定义排序：CORE 最先，ChatCORE 最后，其余在中间
            # Custom ordering: CORE first, ChatCORE last, rest in middle
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # 固定列宽
            # Fixed column widths
            stages = ["base", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # 写入表头
            # Write table header
            header = f"| {'Metric'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # 写入分隔线
            # Write separator
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # 写入表行
            # Write table rows
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # 计算并写入总挂钟时间
            # Calculate and write total wall clock time
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        # 同时将 report.md 复制到当前目录
        # also cp the report.md file to current directory
        print(f"Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """重置报告。
        Reset the report."""
        # 删除章节文件
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # 如果 report.md 存在则删除
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # 生成并写入带有开始时间戳的头部章节
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

# -----------------------------------------------------------------------------
# nanochat 专用的便捷函数
# nanochat-specific convenience functions

class DummyReport:
    """空操作报告对象，用于非 rank 0 进程。"""

    def log(self, *args, **kwargs):
        """空操作日志方法（非 rank 0 进程不记录日志）。"""
        pass

    def reset(self, *args, **kwargs):
        """空操作重置方法（非 rank 0 进程不执行重置）。"""
        pass

def get_report():
    """获取报告对象。仅 rank 0 记录日志，其他 rank 使用 DummyReport。

    Returns:
        对于 rank 0 进程返回 Report 实例，对其他进程返回 DummyReport。
    """
    # 为方便起见，只有 rank 0 记录日志
    # just for convenience, only rank 0 logs to report
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate or reset nanochat training reports.")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="Operation to perform (default: generate)")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
