# NanoChat 项目技术文档

> **项目地址**: https://github.com/karpathy/nanochat
> **作者**: Andrej Karpathy
> **许可证**: MIT
> **Python版本**: >=3.10
> **核心依赖**: PyTorch 2.9.1

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目架构总览](#2-项目架构总览)
3. [从零开始训练指南](#3-从零开始训练指南)
   - [3.1 环境准备](#31-环境准备)
   - [3.2 硬件要求](#32-硬件要求)
   - [3.3 完整训练流程：GPT-2速通](#33-完整训练流程gpt-2速通)
   - [3.4 CPU/MacBook小规模训练](#34-cpumacbook小规模训练)
   - [3.5 与模型对话](#35-与模型对话)
   - [3.6 常见问题排查](#36-常见问题排查)
4. [核心模块详解](#4-核心模块详解)
   - [GPT模型 (gpt.py)](#41-gpt模型-gptpy)
   - [优化器 (optim.py)](#42-优化器-optimpy)
   - [分词器 (tokenizer.py)](#43-分词器-tokenizerpy)
   - [数据加载器 (dataloader.py)](#44-数据加载器-dataloaderpy)
   - [数据集管理 (dataset.py)](#45-数据集管理-datasetpy)
   - [推理引擎 (engine.py)](#46-推理引擎-enginepy)
   - [Flash Attention (flash_attention.py)](#47-flash-attention-flash_attentionpy)
   - [FP8训练 (fp8.py)](#48-fp8训练-fp8py)
   - [检查点管理 (checkpoint_manager.py)](#49-检查点管理-checkpoint_managerpy)
   - [公共工具 (common.py)](#410-公共工具-commonpy)
   - [代码执行沙箱 (execution.py)](#411-代码执行沙箱-executionpy)
   - [损失评估 (loss_eval.py)](#412-损失评估-loss_evalpy)
   - [报告生成 (report.py)](#413-报告生成-reportpy)
5. [训练脚本详解](#5-训练脚本详解)
   - [预训练 (base_train.py)](#51-预训练-base_trainpy)
   - [监督微调 (chat_sft.py)](#52-监督微调-chat_sftpy)
   - [强化学习 (chat_rl.py)](#53-强化学习-chat_rlpy)
   - [评估脚本 (base_eval.py / chat_eval.py)](#54-评估脚本-base_evalpy--chat_evalpy)
   - [Web服务 (chat_web.py)](#55-web服务-chat_webpy)
6. [任务系统 (tasks/)](#6-任务系统-tasks)
7. [运行脚本 (runs/)](#7-运行脚本-runs)
8. [关键技术特性](#8-关键技术特性)
9. [训练流水线](#9-训练流水线)
10. [总结](#10-总结)

---

## 1. 项目概述

NanoChat 是训练大语言模型（LLM）的最小化实验框架。它被设计为在**单个GPU节点**上运行（通常为8×H100），代码精简且高度可修改，覆盖了LLM开发的所有主要阶段：

- **分词**：BPE分词器的训练和评估
- **预训练**：基于GPT架构的基础模型训练
- **微调**：监督微调（SFT）和强化学习（RL）
- **评估**：CORE指标、BPB（bits per byte）、ChatCORE等
- **推理**：支持KV缓存的高效推理引擎
- **对话UI**：类ChatGPT的Web界面

**核心理念**：通过一个单一复杂性调节旋钮 `--depth`（Transformer层数）来确定模型架构，其他超参数（宽度、注意力头数、学习率、训练周期、权重衰减等）均自动推断以达到计算最优。

### 性能数据

| 指标 | 数值 |
|------|------|
| GPT-2级别训练成本（2019年） | ~$43,000（168小时） |
| NanoChat当前训练时间 | ~2小时（8×H100） |
| NanoChat当前训练成本 | ~$48（按需），~$15（竞价实例） |
| GPT-2 CORE基准 | 0.256525 |
| NanoChat d24 CORE | 0.2585+ |

---

## 2. 项目架构总览

```
nanochat/
├── nanochat/                  # 核心库
│   ├── gpt.py                 # GPT Transformer模型定义
│   ├── optim.py               # AdamW + Muon 混合优化器
│   ├── tokenizer.py           # BPE分词器（支持HF和Rust两种实现）
│   ├── dataloader.py          # 分布式数据加载器
│   ├── dataset.py             # 预训练数据集下载与管理
│   ├── engine.py              # 高效推理引擎（KV Cache + 工具调用）
│   ├── flash_attention.py     # Flash Attention 3 / SDPA统一接口
│   ├── fp8.py                 # FP8训练支持（~150行，轻量替代torchao）
│   ├── checkpoint_manager.py  # 模型保存/加载
│   ├── common.py              # 公共工具函数
│   ├── execution.py           # 沙箱化Python代码执行
│   ├── loss_eval.py           # BPB损失评估
│   └── report.py              # 训练报告生成
│
├── scripts/                   # 可执行脚本
│   ├── base_train.py          # 预训练主脚本
│   ├── base_eval.py           # 基础模型评估
│   ├── chat_sft.py            # 监督微调
│   ├── chat_sft_mini.py       # 极简SFT（本地数据，无需HuggingFace）
│   ├── chat_rl.py             # 强化学习（类GRPO）
│   ├── chat_eval.py           # 对话模型评估
│   ├── chat_cli.py            # CLI对话界面
│   ├── chat_web.py            # Web对话服务（FastAPI）
│   ├── tok_train.py           # 分词器训练
│   └── tok_eval.py            # 分词器评估
│
├── tasks/                     # 任务/数据集模块
│   ├── common.py              # Task基类与TaskMixture
│   ├── gsm8k.py               # 数学题（GSM8K）
│   ├── mmlu.py                # 多学科选择题（MMLU）
│   ├── arc.py                 # 科学推理（ARC）
│   ├── humaneval.py           # 代码生成（HumanEval）
│   ├── smoltalk.py            # 通用对话（SmolTalk）
│   ├── spellingbee.py         # 拼写计数任务
│   └── customjson.py          # 自定义JSONL对话数据
│
├── runs/                      # 启动脚本
│   ├── speedrun.sh            # GPT-2速通训练脚本
│   ├── miniseries.sh          # 小系列模型训练
│   ├── scaling_laws.sh        # 缩放定律实验
│   └── runcpu.sh              # CPU/MPS运行示例
│
├── tests/                     # 测试
│   ├── test_engine.py         # 推理引擎测试
│   └── test_attention_fallback.py # 注意力回退测试
│
└── dev/                       # 开发工具
    ├── gen_synthetic_data.py  # 合成身份数据生成
    ├── repackage_data_reference.py # 数据重新打包
    └── scaling_analysis.ipynb # 缩放定律分析
```

---

## 3. 从零开始训练指南

本章提供从零开始训练一个 GPT-2 级别 LLM 的完整、可操作的步骤。涵盖环境配置、数据准备、预训练、微调、推理的全流程。

### 3.1 环境准备

#### 安装 uv（Python 包管理器）

nanochat 使用 [uv](https://docs.astral.sh/uv/) 进行依赖管理：

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 拉取代码并创建虚拟环境

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 创建 Python 虚拟环境（如果 .venv 不存在）
uv venv

# GPU 版本（CUDA，A100/H100 等）
uv sync --extra gpu

# CPU / MPS 版本（MacBook 或 CPU 服务器）
uv sync --extra cpu

# 激活虚拟环境
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 开发者安装（额外安装 pytest、matplotlib、transformers 等）
uv sync --extra gpu --group dev
```

#### 设置 Wandb（推荐但非必须）

```bash
# 安装并登录 wandb
pip install wandb
wandb login

# 如果不想用 wandb，所有脚本都支持 --run=dummy 来跳过日志记录
```

#### 设置环境变量

```bash
# 所有中间产物（数据集、分词器、检查点）的存储目录，默认 ~/.cache/nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# OpenMP 线程数设为 1（避免与 PyTorch 线程冲突）
export OMP_NUM_THREADS=1

# 强制指定计算精度（通常不需要，自动检测即可）
# export NANOCHAT_DTYPE=bfloat16   # 可选：bfloat16 | float16 | float32
```

### 3.2 硬件要求

#### 完整 GPT-2 训练（推荐配置）

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 8×A100 (40/80GB) | 8×H100 (80GB) |
| VRAM | 每GPU至少32GB | 每GPU 80GB |
| 训练时间 | ~4小时 (A100) | ~2小时 (H100) |
| 磁盘空间 | ~120GB 数据集 + ~5GB 检查点 | 同左 |
| 系统内存 | 64GB+ | 128GB+ |
| 成本 | ~$50 (H100 按需) / ~$15 (竞价) | — |

#### 小规模实验 / 学习

| 项目 | 要求 |
|------|------|
| GPU | 单GPU (RTX 3090/4090, A100 等) 或 MPS (Apple Silicon) 或 CPU |
| VRAM | 6GB+ (通过降低 `--device-batch-size` 适应更小显存) |
| 训练时间 | 几分钟到几十分钟（取决于模型大小） |
| 模型规模 | `--depth=4` 到 `--depth=12` |

#### VRAM 不足时的调整策略

如果训练时遇到 CUDA Out of Memory (OOM)：

```bash
# 1. 减小设备批大小（默认 32）
--device-batch-size=16  # 或 8, 4, 2, 1

# 2. 减小序列长度（默认 2048）
--max-seq-len=1024  # 或 512

# 3. 减小模型深度
--depth=12  # 或 8, 6, 4

# 4. 不使用滑动窗口（SDPA回退下窗口注意力性能差）
--window-pattern=L
```

### 3.3 完整训练流程：GPT-2 速通

以下是**端到端训练**的完整流程，对应 `runs/speedrun.sh` 脚本。整个过程在 8×H100 上大约需要 2-3 小时。

本章提供 **Linux** 和 **Windows** 两套命令。主体命令相同（均为 Python 脚本），区别在于 Shell 语法（环境变量设置、后台进程管理等）。

> **通用提示**：建议在后台会话中运行，防止连接断开导致训练中断：
> - Linux：`screen -L -Logfile speedrun.log -S speedrun`
> - Windows：无法使用 screen，建议保持终端窗口开启（或使用 `Start-Process` 后台启动）

---

#### 步骤 1：运行完整速通脚本

**Linux**：

```bash
bash runs/speedrun.sh
```

**Windows (PowerShell)**：

speedrun.sh 全部是 `python` / `torchrun` 命令，只需按顺序逐条执行即可。没有对应的 bash 脚本，直接跳到步骤 2 手动分步执行。

---

#### 步骤 2（分步）：初始化报告

**Linux**：

```bash
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"   # 设置所有中间产物的存储根目录
export OMP_NUM_THREADS=1                            # 限制 OpenMP 线程数，避免与 PyTorch 资源竞争
python -m nanochat.report reset                     # 清空旧报告，写入当前运行环境信息（GPU型号、Git版本等）
```

**Windows (PowerShell)**：

```powershell
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"  # 设置所有中间产物的存储根目录
$env:OMP_NUM_THREADS = "1"                                    # 限制 OpenMP 线程数
python -m nanochat.report reset                                # 清空旧报告，写入当前运行环境信息
```

> **如果想把数据存到其他路径**（如 D 盘）：
> ```powershell
> $env:NANOCHAT_BASE_DIR = "D:\project\python\nanochat\data"
> ```

---

#### 步骤 3（分步）：训练分词器

**Linux**：

```bash
# 下载前 8 个数据分片用于分词器训练（每个分片约 250M 字符，共约 2B 字符）
python -m nanochat.dataset -n 8

# 训练 BPE 分词器（词表大小 32768 = 2^15）
python -m scripts.tok_train

# 评估分词器压缩率（与 GPT-2 和 GPT-4 分词器对比）
python -m scripts.tok_eval
```

**Windows (PowerShell)**：

```powershell
# 下载前 8 个数据分片（每个分片约 250M 字符，共约 2B 字符）
python -m nanochat.dataset -n 8

# 训练 BPE 分词器（词表大小 32768 = 2^15）
python -m scripts.tok_train

# 评估分词器（与 GPT-2 和 GPT-4 对比压缩率）
python -m scripts.tok_eval
```

`tok_train.py` 的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-chars` | 2,000,000,000 | 训练所用最大字符数 |
| `--doc-cap` | 10,000 | 单文档最大字符数（截断） |
| `--vocab-size` | 32768 | 词表大小 |

分词的中间产物保存在 `$env:NANOCHAT_BASE_DIR\tokenizer\` 下：

- `tokenizer.pkl` — 分词器模型
- `token_bytes.pt` — 每个 token 的字节数缓存（用于 bpb 计算）

---

#### 步骤 4（分步）：后台下载更多数据

**Linux**：

```bash
# 启动下载 170 个分片（约 40GB），后台运行
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
```

**Windows (PowerShell)**：

```powershell
# 启动下载 170 个分片（约 40GB），使用 Start-Process 在新窗口中后台运行
Start-Process python -ArgumentList "-m", "nanochat.dataset", "-n", "170" -NoNewWindow

# 注意：Windows 上没有 wait 命令的等价物，
# 可以通过检查 $env:NANOCHAT_BASE_DIR\base_data_climbmix\ 下的文件数量来判断下载是否完成
# 目标：至少 150 个 shard_xxxxx.parquet 文件
```

> **获取下载进度**：
> ```powershell
> # 查看已下载文件数
> (Get-ChildItem "$env:NANOCHAT_BASE_DIR\base_data_climbmix\shard_*.parquet").Count
> ```

数据集分片保存在 `$env:NANOCHAT_BASE_DIR\base_data_climbmix\` 下，文件名为 `shard_00001.parquet` 到 `shard_NNNNN.parquet`。GPT-2 级别训练大约需要 150 个分片。

---

#### 步骤 5（分步）：预训练基础模型

> **前置检查**：确保步骤 4 的下载已完成（分片数 >= 150）。

**Linux**：

```bash
wait $DATASET_DOWNLOAD_PID                              # 等待后台下载完成
# 启动 8 GPU 分布式预训练（核心步骤，耗时~1.5-2小时）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=speedrun
```

**Windows (PowerShell)**：

```powershell
# 启动 8 GPU 分布式预训练（核心步骤，耗时~1.5-2小时）
# torchrun 负责多GPU通信，--standalone 表示单节点模式，--nproc_per_node=8 表示使用8个GPU
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- `
    --depth=24 `                    # 模型层数，d24约等于GPT-2的1.6B参数量
    --target-param-data-ratio=8 `   # Token/参数比，8=略欠训练但要快，12=计算最优
    --device-batch-size=16 `        # 每GPU每步处理16个序列
    --fp8 `                         # 启用FP8混合精度训练加速（仅H100+）
    --run=speedrun                  # wandb运行名称


# 单GPU/低配GPU 快速验证命令（Windows/PowerShell，约15秒跑完）
python -m scripts.base_train `
    --depth=4 `                 # 4层Transformer，约37M参数
    --max-seq-len=256 `         # 序列长度256（完整训练用2048）
    --device-batch-size=1 `     # 每步1个序列（4GB显存也能跑）
    --total-batch-size=512 `    # 全局批次512 token
    --num-iterations=100 `      # 只跑100步验证流程
    --window-pattern=L `        # 全上下文注意力（非H100 GPU必须）
    --core-metric-every=-1 `    # -1=跳过CORE评估（省时间）
    --sample-every=-1 `         # -1=跳过文本采样
    --eval-every=-1 `           # -1=跳过验证集评估
    --run=dummy                 # 不记录wandb日志

```

> **幂等Shell 换行符**：Linux 用 `\`，PowerShell 用 `` ` ``（反引号）。

各参数的含义和调参建议：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth` | 20 | **核心复杂度旋钮**。d24≈GPT-2级别(1.6B参数) |
| `--target-param-data-ratio` | 12 | Token数/参数量比值。8=略欠训练以追赶GPT-2，12=计算最优(Chinchilla≈20) |
| `--device-batch-size` | 32 | 每GPU每步的序列数。**降低此值以节省VRAM**（16→8→4→2→1） |
| `--total-batch-size` | 自动计算 | 全局批次大小（tokens）。-1=根据缩放定律自动计算 |
| `--fp8` | False | 启用FP8训练加速（仅H100+ GPU有效） |
| `--max-seq-len` | 2048 | 最大上下文长度 |
| `--num-iterations` | -1 | 显式指定训练步数（-1=根据参数自动计算） |
| `--run` | dummy | wandb运行名称（`dummy`=跳过wandb日志） |
| `--model-tag` | 自动= `d{深度}` | 模型标识，检查点保存目录名 |
| `--eval-every` | 250 | 每N步评估一次验证bpb |
| `--core-metric-every` | 2000 | 每N步评估CORE指标 |
| `--sample-every` | 2000 | 每N步生成文本样本 |
| `--save-every` | -1 | 每N步保存检查点（-1=仅最后一步保存） |
| `--resume-from-step` | -1 | 从指定步数恢复训练 |
| `--window-pattern` | SSSS | 滑动窗口模式：L=完整上下文，S=1/4上下文，最后一层始终为L |

**训练期间的关键日志解读**：

```
step 00250/07800 (3.21%) | loss: 0.385672 | lrm: 1.00 | dt: 342.15ms | tok/sec: 3,066,667 | bf16_mfu: 48.32 | epoch: 1 pq: 2 rg: 3 | total time: 2.50m | eta: 42.3m
```

| 字段 | 含义 |
|------|------|
| `step` | 当前步数/总步数（完成百分比） |
| `loss` | 去偏 EMA 平滑后的训练损失 |
| `lrm` | 学习率乘数（1.0=预热完成，<1.0=在预热/退火阶段） |
| `dt` | 单步耗时 |
| `tok/sec` | 每秒处理的 token 数（全节点） |
| `bf16_mfu` | BF16 模型 FLOPs 利用率（H100 峰值~989TFlops/GPU） |
| `epoch / pq / rg` | 数据遍历状态：第几个epoch，第几个parquet文件，第几个row group |
| `total time / eta` | 累计训练时间 / 预估剩余时间 |

**检查点结构**（保存在 `$env:NANOCHAT_BASE_DIR\base_checkpoints\d24\`）：

```
model_007800.pt       # 模型参数
optim_007800_rank0.pt # 优化器状态（每 rank 一个文件）
optim_007800_rank1.pt
...
meta_007800.json       # 元数据（配置、数据加载器状态、循环状态）
```

---

#### 步骤 6（分步）：评估基础模型

**Linux**：

```bash
# 多GPU评估基础模型：CORE指标 + BPB + 文本生成样本
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
```

**Windows (PowerShell)**：

```powershell
# 多GPU评估基础模型：CORE指标 + BPB + 文本生成样本
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16
```

评估内容：

- **CORE 指标**（DCLM 标准评估）
- **BPB**（bits per byte，验证集和训练集）
- **文本生成样本**（多个提示词的补全）

---

#### 步骤 7（分步）：监督微调（SFT）

**Linux**：

```bash
# 下载身份对话数据：给模型赋予"个性"（名字、爱好、创作者等）
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 多GPU监督微调：让模型学会对话格式、工具调用、多选题等能力
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 --run=speedrun

# 多GPU评估对话模型：ChatCORE 指标（ARC, MMLU, GSM8K, HumanEval, SpellingBee）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Windows (PowerShell)**：

```powershell
# 下载身份对话数据（约 2.3MB）：给模型赋予"个性"
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" `
    -OutFile "$env:NANOCHAT_BASE_DIR\identity_conversations.jsonl"

# 多GPU监督微调：让模型学会对话格式、工具调用、多选题等能力
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- `
    --device-batch-size=16 `
    --run=speedrun

# 多GPU评估对话模型：ChatCORE 指标
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

SFT 训练混合了以下数据（详见 `scripts/chat_sft.py`）：

| 数据集 | 轮数 | 用途 |
|--------|------|------|
| SmolTalk (460K) | 1 | 通用对话能力 |
| Identity JSONL (1K) | 2 | 模型个性 (名字、爱好、创作者等) |
| MMLU (100K) | 3 | 多选题格式 |
| GSM8K (8K) | 4 | 数学推理 + 工具调用 |
| SimpleSpelling (200K) | 1 | 拼写单词 |
| SpellingBee (80K) | 1 | 数字母个数 |

---

#### 步骤 8（分步）：生成训练报告

**Linux**：

```bash
# 汇总所有阶段的日志，生成完整训练报告（report.md）
python -m nanochat.report generate
```

**Windows (PowerShell)**：

```powershell
# 汇总所有阶段的日志，生成完整训练报告（report.md）
python -m nanochat.report generate
```

生成的文件路径：`$env:NANOCHAT_BASE_DIR\report\report.md`（并复制到当前目录），内容包含：

- 分词器压缩率对比 (GPT-2 vs GPT-4 vs Ours)
- 预训练 CORE 指标和 BPB
- SFT 后 ChatCORE（ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee）
- 各阶段训练时间、算力消耗、成本估算

---

### 3.4 CPU/MacBook 小规模训练

如果只有 MacBook 或 CPU 服务器，可以运行 `runs/runcpu.sh` 体验完整流程，但模型能力会远弱于 GPU 版本。

```bash
# 完整流程（约 40 分钟，M3 Max MacBook Pro）
bash runs/runcpu.sh
```

或分步执行：

```bash
# 1. 环境配置
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"  # 数据存储目录
mkdir -p $NANOCHAT_BASE_DIR                        # 确保目录存在
uv sync --extra cpu                                # 安装CPU版PyTorch
source .venv/bin/activate                          # 激活虚拟环境

# 2. 分词器训练（约 34 秒，M3 Max）
python -m nanochat.dataset -n 8                     # 下载8个数据分片（约800MB）
python -m scripts.tok_train --max-chars=2000000000  # 用2B字符训练分词器
python -m scripts.tok_eval                           # 评估分词器压缩率

# 3. 训练小型模型（约 30 分钟，M3 Max）
python -m scripts.base_train \
    --depth=6 \              # 仅 6 层（完整训练用 d24）
    --head-dim=64 \          # 注意力头维度 64（默认128）
    --window-pattern=L \     # 全上下文注意力（SDPA不支持滑动窗口）
    --max-seq-len=512 \      # 短序列（完整训练用2048）
    --device-batch-size=32 \ # 每步序列数
    --total-batch-size=16384 \ # 全局批次大小
    --eval-every=100 \       # 每100步评估验证损失
    --eval-tokens=524288 \   # 评估用token数
    --core-metric-every=-1 \ # -1=训练期间跳过CORE评估（省时间）
    --sample-every=100 \     # 每100步生成文本样本
    --num-iterations=5000 \  # 训练步数
    --run=dummy              # 跳过wandb日志

# 4. 基础模型评估：CORE指标 + BPB + 文本样本
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# 5. SFT 监督微调（约 10 分钟，M3 Max）
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=dummy
```

在 M3 Max MacBook Pro 上的测试结果：d6、head_dim=64 的小模型 SFT 后能正确回答巴黎是法国首都，但能力非常有限——仅为教学/实验用途。

---

### 3.5 与模型对话

训练完成后，有两种方式与模型交互。

#### 命令行对话（chat_cli.py）

```bash
# 单次问答模式：模型回答后自动退出
python -m scripts.chat_cli -p "为什么天空是蓝色的？"

# 交互式对话模式：持续对话直到输入 quit/exit
python -m scripts.chat_cli

# 关键参数说明：
#   -i sft|rl      模型训练阶段：sft=监督微调，rl=强化学习（默认 sft）
#   -g d24         模型标识（对应 base_checkpoints/d24）
#   -s 7800        检查点步数（不指定则自动取最新）
#   -t 0.8         采样温度，0=贪婪解码，越高越随机
#   -k 50          top-k 采样，0=使用全部词表
#   --device-type  设备类型 cuda|cpu|mps（不指定则自动检测）
```

交互命令：
- 输入 `quit` 或 `exit` 退出
- 输入 `clear` 开始新对话
- 按 `Ctrl+C` 退出

#### Web 对话界面（chat_web.py）

```bash
# 启动 Web 服务（单 GPU）—— 启动后浏览器访问 http://localhost:8000
python -m scripts.chat_web

# 多 GPU 数据并行 —— 每个 GPU 加载一个完整模型副本，请求自动分发
python -m scripts.chat_web --num-gpus 4

# 完整参数示例
python -m scripts.chat_web -i sft -g d24 -p 8000

# 关键参数说明：
#   -n, --num-gpus    GPU 数量（默认 1），多GPU实现数据并行
#   -i, --source      模型来源：sft|rl（默认 sft）
#   -g, --model-tag   模型标识（如 d24），不指定则用最大的模型
#   -s, --step        检查点步数（不指定则自动取最新）
#   -p, --port        服务端口（默认 8000）
#   -t, --temperature API默认采样温度（0.8）
#   -k, --top-k       API默认 top-k 采样（50）
#   -m, --max-tokens  API默认最大生成 token 数（512）
#   --host            绑定地址（默认 0.0.0.0，允许外部访问）
```

启动后打开浏览器访问 `http://localhost:8000`。

**云服务器注意事项**：如果在云 GPU 实例上运行，需要使用实例的**公网 IP** 来访问，例如 `http://209.20.xxx.xxx:8000/`。请确保安全组/防火墙已放行对应端口。

**Web 界面的滥用防护**（自动生效）：

| 限制项 | 上限 |
|--------|------|
| 每条请求最大消息数 | 500 |
| 每条消息最大字符数 | 8,000 |
| 总对话最大字符数 | 32,000 |
| Temperature 范围 | 0.0 - 2.0 |
| Top-k 范围 | 0 - 200 |
| Max tokens 范围 | 1 - 4,096 |

**API 端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天界面 HTML |
| POST | `/chat/completions` | SSE 流式聊天 API |
| GET | `/health` | 健康检查 + Worker 池状态 |
| GET | `/stats` | Worker 池详细统计 |
| GET | `/logo.svg` | Logo 文件 |

---

### 3.6 常见问题排查

#### Q1: CUDA Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**解决方案**（按优先级排列）：

1. 减小 `--device-batch-size`：`32 → 16 → 8 → 4 → 2 → 1`
2. 减小 `--max-seq-len`：`2048 → 1024 → 512`
3. 减小 `--depth`：`24 → 20 → 16 → 12`
4. 不使用 FP8（如果 VRAM 非常紧张）：去掉 `--fp8` 参数

#### Q2: 只有一个 GPU

去掉 `torchrun`，直接运行 Python 脚本。代码会自动使用梯度累积来模拟大批次：

```bash
# 单GPU训练：去掉torchrun，直接用python启动，梯度累积自动模拟大批次
python -m scripts.base_train --depth=24 --device-batch-size=16
```

注意：单个 GPU 需要约 8 倍的训练时间。

#### Q3: 恢复中断的训练

```bash
# 从第 5000 步恢复训练：加载模型+优化器状态+数据加载器位置，无缝接续
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --resume-from-step=5000 --run=speedrun
```

恢复时会自动：
- 加载模型参数和优化器状态
- 从上次的数据加载器位置继续
- 恢复 `min_val_bpb`, `smooth_train_loss`, `total_training_time` 等循环状态

#### Q4: SDPA 回退 + 滑动窗口导致性能极差

在非 Hopper GPU 上，Flash Attention 3 不可用，会回退到 PyTorch SDPA。SDPA **不支持滑动窗口**注意力，会导致性能显著下降。

**解决方案**：添加 `--window-pattern=L` 使用全上下文注意力：

```bash
# 强制全上下文注意力：SDPA回退时禁用滑动窗口，避免性能严重下降
python -m scripts.base_train --depth=24 --window-pattern=L
```

#### Q5: 分词器与模型词表不匹配

```
AssertionError: Tokenizer vocab size X does not match model config vocab size Y
```

这意味着使用的分词器和模型检查点不匹配。解决方案：
- 删除 `$NANOCHAT_BASE_DIR/tokenizer/` 并重新训练分词器
- 或使用与检查点同批次训练的分词器

#### Q6: 数据集升级（FinewebEdu → ClimbMix）

2026年3月4日后进行 `git pull` 的用户会看到数据集升级警告：

```
WARNING: DATASET UPGRADE REQUIRED
nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.
```

执行以下命令完成升级：

```bash
python -m nanochat.dataset -n 170     # 重新下载170个ClimbMix分片（约40GB）
python -m scripts.tok_train           # 基于新数据集重新训练BPE分词器
```

#### Q7: uv 同步失败

```bash
# 确保 uv 是最新版本
uv self update

# 清理缓存重试
uv cache clean
uv sync --extra gpu

# 如果特定包安装失败，检查 Python 版本 >=3.10
python --version
```

#### Q8: fp16 训练出现 NaN / 梯度下溢

fp16 的指数范围有限，数值稳定性较差。建议：

```bash
# 方案1：使用 bf16（H100/A100 推荐）
export NANOCHAT_DTYPE=bfloat16

# 方案2：使用 fp32（最稳定，但最慢）
export NANOCHAT_DTYPE=float32
```

#### Q9: HuggingFace 连不上（国内网络）

SFT 需要从 HuggingFace 下载 SmolTalk、GSM8K、MMLU 等数据集，墙内无法直接访问。解决方案是使用项目内置的 **极简 SFT 脚本**，只用本地身份 JSONL 数据：

```bash
# Linux
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft_mini --num-iterations=500 --device-batch-size=16

# Windows PowerShell
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" `
    -OutFile "$env:NANOCHAT_BASE_DIR\identity_conversations.jsonl"
python -m scripts.chat_sft_mini --num-iterations=500 --device-batch-size=16

# Windows cmd
curl -L -o "%NANOCHAT_BASE_DIR%\identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft_mini --num-iterations=500 --device-batch-size=16
```

> `chat_sft_mini.py` 的作用：只加载本地的 identity JSONL，不连 HuggingFace。用 1000 条身份对话数据做 SFT，教会模型基本的对话格式。适合验证流程和网络受限场景，但效果不如完整 SFT（缺少 SmolTalk 的 46 万条通用对话数据）。

`chat_sft_mini.py` 参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-iterations` | 500 | SFT 训练步数 |
| `--device-batch-size` | 16 | 每步序列数 |
| `--max-seq-len` | 512 | 最大上下文长度 |
| `--lr` | 1e-4 | 学习率 |

---

## 4. 核心模块详解

### 4.1 GPT模型 (gpt.py)

GPT模型的实现文件，约513行，是项目的核心。继承自 `nn.Module`。

**架构特点**：

| 特性 | 说明 |
|------|------|
| 位置编码 | Rotary Embeddings（旋转位置编码），无学习参数 |
| 注意力机制 | 因果自注意力 + QK归一化 + 滑动窗口注意力 + GQA |
| 激活函数 | ReLU²（`relu(x).square()`） |
| 归一化 | RMSNorm（无可学习参数） |
| 线性层 | 自定义Linear（权重FP32存储，前向时转为计算精度） |
| 词嵌入/LM Head | 不绑定权重（untied） |
| 偏置 | 所有线性层无偏置 |
| 值嵌入 | ResFormer风格的Value Embedding（交替层启用） |
| 逐层缩放 | resid_lambdas（残差缩放）+ x0_lambdas（初始嵌入混合） |
| Smear机制 | 混合前一位置的嵌入到当前位置（廉价的双元组信息） |
| Backout机制 | 从中层减去残差缓存以移除底层特征 |
| Logit软截断 | softcap=15的tanh截断 |

**GPTConfig 配置类**：

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 最大序列长度
    vocab_size: int = 32768       # 词表大小 (2^15)
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query注意力头数
    n_kv_head: int = 6            # KV注意力头数（GQA支持）
    n_embd: int = 768             # 嵌入维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

**核心类结构**：

- `GPT` - 主模型类，包含embedding、Transformer blocks、lm_head、各种缩放参数、值嵌入层
- `CausalSelfAttention` - 因果自注意力层，支持GQA、Rotary、QK Norm、滑动窗口
- `MLP` - 前馈网络，使用ReLU²激活
- `Block` - Transformer的基本构建块（Attention + MLP）
- `Linear` - 自定义线性层，自动将权重转换为输入精度

**前向传播流程**：

1. 词嵌入 → 归一化
2. Smear操作（混合前一位置嵌入）
3. 逐层处理（每个Block包含 Attention + MLP，使用pre-norm结构）
4. 残差缩放和初始嵌入混合（x0_lambdas, resid_lambdas）
5. Backout减法
6. 最终归一化 → LM Head → Logit软截断
7. 交叉熵损失计算（训练时）

**生成方法** (`generate`)：

- 简单的自回归逐Token生成
- 支持temperature采样和top-k过滤
- 返回Python生成器，流式输出token

### 4.2 优化器 (optim.py)

约536行，实现了**混合优化器策略**：

- **Muon**: 用于2D矩阵参数（Transformer的线性层权重），基于动量 + 正交化
- **AdamW**: 用于嵌入层、标量参数、LM Head

**两种实现**：

| 类名 | 用途 |
|------|------|
| `MuonAdamW` | 单GPU版本，用于参考和测试 |
| `DistMuonAdamW` | 分布式版本，支持ZeRO-2风格的状态分片 |

**Muon优化的关键特性**：

1. **Nesterov动量** - 在正交化前先应用动量更新
2. **Polar Express正交化** - 替代传统的Newton-Schulz迭代，具有更好的收敛特性
3. **NorMuon方差衰减** - 每个神经元/列的自适应学习率，归一化正交化后的更新尺度
4. **谨慎权重衰减** - 仅当梯度方向与参数方向一致时应用权重衰减
5. **Fused Kernel** - 使用 `torch.compile` 将动量→正交化→方差衰减→更新的完整流程编译为单个CUDA kernel

**DistMuonAdamW的通信模式**（3阶段异步）：

- 阶段1：启动所有异步reduce操作
- 阶段2：等待reduce → 计算更新 → 启动gather
- 阶段3：等待gather → 拷贝回原始参数

**优化器参数分组**（`GPT.setup_optimizer`）：

| 参数组 | 优化器 | 学习率 | weight_decay | 备注 |
|--------|--------|--------|-------------|------|
| lm_head | AdamW | 0.004×scale | 0.01 | |
| embedding | AdamW | 0.2×scale | 0.001 | |
| value_embeds | AdamW | 0.1×scale | 0.01 | |
| resid_lambdas | AdamW | 0.005 | 0.05 | |
| x0_lambdas | AdamW | 0.5 | 0.0 | beta1=0.96 |
| smear相关 | AdamW | 0.2 | 0.0 | |
| 矩阵参数 | Muon | 0.02×scale | 可调 | 按shape分组 |

### 4.3 分词器 (tokenizer.py)

约407行，实现了GPT-4风格的BPE分词器。

**两种实现**：

| 实现 | 训练引擎 | 推理引擎 | 优势 |
|------|---------|---------|------|
| `HuggingFaceTokenizer` | HuggingFace BPE | HuggingFace | 易于使用，生态兼容 |
| `RustBPETokenizer` | rustbpe（Rust） | tiktoken（高效） | 训练更快，推理效率高 |

**特殊Token**：

```
<|bos|>              - 文档开始标记
<|user_start|>       - 用户消息开始
<|user_end|>         - 用户消息结束
<|assistant_start|>  - 助手消息开始
<|assistant_end|>    - 助手消息结束
<|python_start|>     - Python工具调用开始
<|python_end|>       - Python工具调用结束
<|output_start|>     - 工具输出开始
<|output_end|>       - 工具输出结束
```

**关键功能**：

- `render_conversation()` - 将对话渲染为token序列和loss mask（只对assistant回复计算loss）
- `render_for_completion()` - RL模式下使用，删除最后一条assistant消息，添加assistant_start作为提示

### 4.4 数据加载器 (dataloader.py)

约167行，实现**BOS对齐的Best-Fit裁剪**数据加载策略。

**算法**：

1. 每行以BOS token开始
2. 从缓冲区查找能完全放入剩余空间的最大文档
3. 当没有文档能完全放入时，裁剪最短文档填充剩余空间
4. 100%利用率（无padding），约35%的token因裁剪而丢弃

**分布式支持**：

- DDP分片：各rank处理不同的row group
- 断点续训：通过 `(pq_idx, rg_idx, epoch)` 三元组跟踪位置

### 4.5 数据集管理 (dataset.py)

约166行，管理预训练数据集的下载和访问。

**当前数据集**: [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)（由HuggingFace托管）

**关键特性**：

- 多进程并行下载（默认4个工作进程）
- 断点续传（带指数退避重试）
- 自动识别train/val分片（最后一个分片为验证集）
- 向后兼容FinewebEdu-100B数据集

### 4.6 推理引擎 (engine.py)

约358行，实现高效的批量推理。

**KVCache类**：

- 为Flash Attention 3 API设计（`(B, T, H, D)` 格式）
- 支持prefill → 批量decode的高效流程
- 跟踪每个批次元素的序列长度

**Engine.generate() 核心流程**：

1. Batch=1的prefill（填充KV缓存）
2. 将KV缓存复制到num_samples份
3. 逐token生成，支持：
   - 工具调用状态机（Python计算器）
   - 强制token注入（工具输出）
   - 流式yield（`token_column, token_masks`）
4. 终端条件：`<|assistant_end|>` 或 `<|bos|>` token

**计算器工具** (`use_calculator`)：

- 安全地评估Python数学表达式
- 支持字符串操作（如 `.count()`）
- 包含超时机制和危险模式过滤

### 4.7 Flash Attention (flash_attention.py)

约188行，提供统一的Flash Attention接口，自动选择最佳实现。

**自动选择策略**：

| GPU | 计算精度 | 使用实现 |
|-----|---------|---------|
| Hopper (SM90) | bfloat16 | Flash Attention 3（来自`kernels`包） |
| 其他GPU / fp16 / fp32 | 任意 | PyTorch SDPA回退 |

**导出API**（与FA3接口一致）：

```python
flash_attn.flash_attn_func(q, k, v, causal, window_size)      # 训练
flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, ...)   # 推理
```

**SDPA回退处理**：

- 支持滑动窗口注意力（通过显式mask构建）
- 支持GQA（通过SDPA的 `enable_gqa` 参数）
- 支持chunk推理（预填充长度与缓存长度不同）

### 4.8 FP8训练 (fp8.py)

约267行，极小化的FP8训练实现。比torchao的Float8Linear（~2000行）更轻量。

**核心思想**：

- 使用 `torch._scaled_mm`（cuBLAS FP8 kernel）进行矩阵乘法
- Tensorwise动态缩放（每个tensor一个scale）
- 自定义 `autograd.Function` 包装三个GEMM（前向1个+反向2个）

**FP8格式选择**：

| 格式 | 用途 | 范围 |
|------|------|------|
| `float8_e4m3fn` | 输入和权重（更高精度） | [-448, 448] |
| `float8_e5m2` | 梯度（更宽范围） | [-57344, 57344] |

**与torchao的区别**：

| 方面 | NanoChat FP8 | torchao |
|------|-------------|---------|
| 代码行数 | ~150 | ~2000 |
| 实现方式 | 单个autograd.Function | Tensor子类 + __torch_dispatch__ |
| torch.compile | 单个不透明操作 | 可分解到每个op |
| 融合能力 | 边界不可融合 | Inductor可跨边界融合 |

### 4.9 检查点管理 (checkpoint_manager.py)

约195行，管理模型的保存和加载。

**文件命名**：

- `model_{step:06d}.pt` - 模型参数
- `optim_{step:06d}_rank{rank}.pt` - 优化器状态（分片）
- `meta_{step:06d}.json` - 元数据

**关键功能**：

- `save_checkpoint()` / `load_checkpoint()` - 保存/加载检查点
- `build_model()` - 构建模型（支持meta device，避免临时内存分配）
- `find_largest_model()` - 自动选择最大模型
- `load_model()` - 便捷加载（支持base/sft/rl三种来源）
- `_patch_missing_config_keys()` / `_patch_missing_keys()` - 旧检查点向后兼容

### 4.10 公共工具 (common.py)

约279行，提供全局公共功能。

**关键特性**：

| 功能 | 说明 |
|------|------|
| `COMPUTE_DTYPE` | 全局计算精度（自动检测或通过`NANOCHAT_DTYPE`环境变量设置） |
| `compute_init()` | 标准初始化：设备检测、随机种子、精度设置、DDP初始化 |
| `compute_cleanup()` | 清理DDP进程组 |
| `get_peak_flops()` | GPU峰值FLOPS查询（支持NVIDIA Blackwell/Hopper/Ampere/Ada/RTX和AMD CDNA） |
| `download_file_with_lock()` | 文件下载（带文件锁防止并发） |
| `DummyWandb` | wandb的占位实现 |
| `print0()` | 仅rank 0打印 |
| `ColoredFormatter` | 彩色日志格式化 |

**自动精度检测**：

| 硬件 | 默认精度 | 原因 |
|------|---------|------|
| CUDA SM 80+ (A100, H100) | bfloat16 | 原生bf16 Tensor Core |
| CUDA SM < 80 (V100, T4) | float32 | 无bf16支持 |
| CPU / MPS | float32 | 无低精度Tensor Core |

### 4.11 代码执行沙箱 (execution.py)

约350行，安全的Python代码执行环境，改编自OpenAI HumanEval。

**安全措施**：

- 独立进程执行（可超时终止）
- 内存限制（默认256MB）
- stdout/stderr捕获
- 临时目录执行（结束后删除）
- 禁用危险函数（os.system, os.kill, shutil.rmtree, subprocess.Popen等）
- 禁用危险模块（ipdb, psutil等）

**限制说明**（非安全沙箱）：

- 不阻止网络访问
- 无内核级隔离（无seccomp/容器/虚拟化）
- ctypes等动态特性可能绕过限制

### 4.12 损失评估 (loss_eval.py)

约66行，实现**bits per byte (bpb)** 评估指标。

**BPB的优势**：

- 与词表大小无关（即使改变词表大小也能公平对比）
- 按token的字节数加权（特殊token贡献0字节）
- 支持分布式求和归约

### 4.13 报告生成 (report.py)

约420行，自动生成训练报告。

**报告结构**：

1. Header：Git信息、硬件信息、系统信息、代码统计（Bloat metrics）
2. 各阶段报告：分词器训练/评估、预训练/评估、SFT/评估、RL/评估
3. Summary表格：各阶段的CORE、ChatCORE等指标对比
4. 总耗时

---

## 5. 训练脚本详解

### 5.1 预训练 (base_train.py)

约631行，最核心的训练脚本。

**命令行参数分类**：

| 类别 | 关键参数 | 默认值 |
|------|---------|--------|
| 模型架构 | `--depth`, `--aspect-ratio`, `--head-dim`, `--max-seq-len` | 20, 64, 128, 2048 |
| 训练周期 | `--num-iterations`, `--target-flops`, `--target-param-data-ratio` | -1, -1, 12 |
| 优化 | `--device-batch-size`, `--embedding-lr`, `--matrix-lr`, `--weight-decay` | 32, 0.3, 0.02, 0.28 |
| 评估 | `--eval-every`, `--core-metric-every`, `--sample-every`, `--save-every` | 250, 2000, 2000, -1 |
| FP8 | `--fp8`, `--fp8-recipe` | False, "tensorwise" |

**缩放定律自动推导**（muP风格）：

1. **最优训练Token数**：`target_tokens = target_param_data_ratio × scaling_params`
2. **最优批大小**：`Batch ∝ D^0.383`（基于Power Lines论文）
3. **学习率缩放**：`η ∝ √(Batch/Batch_ref)`（AdamW/Muon标准缩放）
4. **权重衰减缩放**：基于 `T_epoch` 恒常框架

**学习率调度**：

- 线性预热（默认40步）
- 常数阶段
- 线性预热退火（默认占总步数65%）
- 最终LR为初始LR的5%

**Muon动量调度**：

- 预热：0.85 → 0.97（前400步）
- 稳定：0.97
- 预热退火：0.97 → 0.90

**训练循环关键操作**：

- 梯度累积实现大批次训练
- FP8仅在训练时启用，评估时切换回BF16（通过`disable_fp8`上下文管理器）
- GC管理：初始化后`gc.freeze()`，完全禁用GC，每5000步手动回收一次

### 5.2 监督微调 (chat_sft.py)

约520行，在预训练模型基础上进行对话能力微调。

**训练数据混合**：

| 数据集 | 轮数 | 数量 | 目的 |
|--------|------|------|------|
| SmolTalk | 1 | 460K | 通用对话能力 |
| CustomJSON (identity) | 2 | 1K+1K | 个性注入 |
| MMLU | 3 | 100K×3 | 多选题能力 |
| GSM8K | 4 | 8K×4 | 数学和工具使用 |
| SimpleSpelling | 1 | 200K | 拼写能力 |
| SpellingBee | 1 | 80K | 字母计数能力 |

**关键设计差异**（与预训练对比）：

- 从预训练检查点继承优化器状态（动量缓冲区热启动）
- 使用progress（0→1）而非绝对步数进行LR调度
- 使用padding而非裁剪（不丢弃任何token）
- Loss mask确保只对assistant内容计算损失
- `last_step` 通过分布式all_reduce确保所有rank同步停止

### 5.3 强化学习 (chat_rl.py)

约333行，简化的GRPO（Group Relative Policy Optimization）实现。

**"简化版GRPO"的含义**：

1. 无信任域（无KL正则化到参考模型）
2. On-policy（无需PPO ratio+clip）
3. DAPO风格的token级归一化
4. 只使用 `(reward - mean)` 作为优势（而非z-score归一化）

**训练流程**：

1. 从GSM8K获取问题
2. 为每个问题生成 `num_samples` 个回答
3. 计算每个回答的reward（答案正确=1，错误=0）
4. 计算优势：`advantages = rewards - mean(rewards)`
5. 梯度更新：最大化 `log_prob × advantage`

### 5.4 评估脚本 (base_eval.py / chat_eval.py)

**基础模型评估**：

- CORE指标（DCLM论文的标准评估）
- train/val BPB
- 采样生成

**对话模型评估（ChatCORE）**：

- 6个任务：ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee
- 分类任务使用centered accuracy：`(acc - baseline) / (1 - baseline)`
- 生成任务使用原始accuracy
- 最终ChatCORE = 所有任务centered accuracy的均值

### 5.5 Web服务 (chat_web.py)

约408行，基于FastAPI的ChatGPT式Web界面。

**架构特点**：

- **数据并行多GPU**：每个GPU加载完整模型副本
- **Worker Pool模式**：请求通过 `asyncio.Queue` 分发到可用worker
- **流式响应**：SSE（Server-Sent Events）协议
- **UTF-8安全**：延迟解码策略确保emoji等多字节字符正确输出

**API端点**：

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 聊天UI页面 |
| GET | `/logo.svg` | Logo文件 |
| POST | `/chat/completions` | 聊天补全（仅流式） |
| GET | `/health` | 健康检查 |
| GET | `/stats` | Worker池统计 |

**滥用防护**：

- 每条请求最多500条消息
- 每条消息最多8000字符
- 总对话不超过32000字符
- Temperature限制在0.0-2.0
- Top-k限制在0-200
- Max tokens限制在1-4096

---

## 6. 任务系统 (tasks/)

### 基类设计

`Task` 基类支持轻量级逻辑切片（`start`, `stop`, `step`），允许在不复制数据的情况下创建数据子集。

**核心属性/方法**：

- `eval_type`：`'generative'` 或 `'categorical'`
- `get_example(index)`：返回对话格式的数据
- `evaluate(conversation, completion)`：评估回答正确性
- `reward(conversation, completion)`：用于RL的奖励函数

**TaskMixture**：多个Task的随机混合（确定性shuffle，seed=42），支持过采样（重复传递同一Task）

**对话格式**（统一格式）：

```python
{
    "messages": [
        {"role": "user", "content": "问题文本"},
        {"role": "assistant", "content": "回答文本" | [{"type": "text", "text": "..."}, {"type": "python", "text": "expr"}]}
    ]
}
```

### 各任务说明

| 任务 | eval_type | 评估方式 | 特殊功能 |
|------|-----------|---------|---------|
| GSM8K | generative | 提取`####`后的数字答案比对 | 工具调用解析（`<<>>`标记） |
| MMLU | categorical | 比较选项字母（A/B/C/D） | 多学科，57个子集 |
| ARC | categorical | 比较选项字母 | 科学推理（Easy/Challenge） |
| HumanEval | generative | 代码执行验证测试 | 沙箱执行 |
| SmolTalk | - | （仅训练） | 46万条通用对话 |
| SpellingBee | generative | 精确字符串匹配 | 字母计数 |
| SimpleSpelling | generative | 精确字符串匹配 | 单词拼写 |

---

## 7. 运行脚本 (runs/)

### speedrun.sh — GPT-2速通

完整的端到端训练流水线（~3小时）：

1. 安装依赖（uv venv + sync）
2. 训练分词器（下载8个数据分片，训练32K BPE）
3. 后台下载170个数据分片
4. d24预训练（fp8，D:P=8）
5. 基础模型评估
6. 下载身份对话数据
7. SFT训练
8. 对话模型评估
9. 生成训练报告

### scaling_laws.sh — 缩放定律实验

在不同模型深度上运行短时训练，分析token-to-params比、训练时长等关系。

### miniseries.sh — 小系列模型

在多个深度（d3, d6, d12, d20, d26）上训练计算最优模型系列。

---

## 8. 关键技术特性

### 8.1 滑动窗口注意力

通过 `window_pattern` 配置：
- `L`：完整上下文（序列长度）
- `S`：短窗口（序列长度/4，向上取整到FA3 tile边界）
- 最后一层始终使用完整上下文

默认模式 `SSSL` 意味着大多数层使用短窗口（节省计算），间隔出现完整上下文层（保持全局信息流）。

### 8.2 Value Embedding (ResFormer)

交替层注入值嵌入，增强位置感知能力：

- 通过 `has_ve(layer_idx, n_layer)` 决定是否启用
- 最后一层始终启用
- 门控机制：`gate = 3 × sigmoid(ve_gate(x[:12]))`，范围(0, 3)

### 8.3 Smear和Backout

**Smear** — 廉价的双元组信息注入：

- 将前一位置的嵌入与当前位置混合
- 可学习门控参数 `smear_lambda`
- 训练时使用快速切片，推理时使用KV缓存存储前一嵌入

**Backout** — 移除底层特征：

- 从中层（n_layer // 2）缓存残差
- 在最终归一化前从最终残差中减去
- 可学习参数 `backout_lambda`

### 8.4 逐层缩放

- `resid_lambdas`：控制每层残差流的缩放（从1.15线性降至1.05）
- `x0_lambdas`：控制每层混入初始嵌入的比例（从0.20线性降至0.05）
- 早起层获得更强的残差和更多的初始嵌入混合

### 8.5 GQA (Group Query Attention)

通过 `n_kv_head < n_head` 实现：

- 减少KV缓存的内存占用（推理时）
- 通过 `SDPA` 的 `enable_gqa` 参数自动处理

### 8.6 混合精度策略

不使用 `torch.amp.autocast`，而是：

- 模型权重存储在FP32（优化器精度）
- 自定义 `Linear` 层在前向时将权重转换为 `COMPUTE_DTYPE`
- 嵌入直接存储在 `COMPUTE_DTYPE` 中
- 通过 `NANOCHAT_DTYPE` 环境变量可覆盖默认精度

---

## 9. 训练流水线

```
                   数据下载
                      │
              ┌───────▼───────┐
              │  训练分词器    │  (tok_train.py)
              │  BPE, 32K词表  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  评估分词器    │  (tok_eval.py)
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │   预训练       │  (base_train.py)
              │  8×H100节点    │
              │  d24+d26       │
              │  FP8+Batch     │
              │  ~1.5-2小时    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  评估基础模型   │  (base_eval.py)
              │  CORE + BPB    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  SFT训练       │  (chat_sft.py)
              │  对话+工具+MC   │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  评估对话模型   │  (chat_eval.py)
              │  ChatCORE      │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  RL训练(可选)   │  (chat_rl.py)
              │  GSM8K GRPO    │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │  Web/CLI对话   │  (chat_web.py / chat_cli.py)
              └───────────────┘
```

---

## 10. 总结

NanoChat 是一个设计哲学非常清晰的项目——**极简、可修改、端到端**。它避免了工业级LLM框架的过度抽象和配置复杂性，用最少量的代码（约7000行纯Python）展示了一个完整的ChatGPT流水线。

**关键优势**：

1. **单一复杂度旋钮**：`--depth` 决定一切，自动推导最优超参数
2. **最先进的训练技术**：FP8训练、Muon优化器、Polar Express正交化、Flash Attention 3
3. **极致的计算效率**：在8×H100上2小时达到GPT-2水平（成本<$50）
4. **完整的LLM生命周期**：分词→预训练→SFT→RL→部署
5. **代码质量**：清晰的结构、详细的注释、精致的实现细节

**适用场景**：

- LLM训练的学习和研究
- 快速原型验证新想法
- 微调能力的实验
- 教学和演示用途
