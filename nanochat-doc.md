# nanochat 笔记：`.vscode/launch.json` 全链路解析

> 本文档对 `.vscode/launch.json` 里每个调试配置实际执行的命令做"刨根问底"式笔记：
> 从入口脚本出发，**递归追踪它 import 的每个模块**，一直追到第三方库边界；
> 记录全流程**张量/矩阵形状**（以 launch.json 实际参数推导出的 **d2 模型** 为具体示例）；
> 凡是能图解的地方一律用 ASCII 图展示。
>
> 事实依据（均已实地核实）：
> - 入口：`.vscode/launch.json`（8 个配置 → 6 个脚本）
> - 本项目代码：`scripts/`（6 个入口）、`nanochat/`（13 个模块）、`tasks/`（2 个）
> - 实际运行产物：`data/tokenizer/`、`data/base_checkpoints/d2/model_000100.pt`、
>   `data/chatsft_checkpoints/d2/model_000200.pt`、`data/eval_bundle/`、`data/report/`
> - 关键元数据：`data/base_checkpoints/d2/meta_000100.json` → `vocab_size=16384, n_layer=2, n_embd=128, n_head=1`

---

## 0. 一页速览

| launch 配置 | 入口脚本 | 核心参数 | 主要产物 / 输出 |
|---|---|---|---|
| `1b. Tokenizer Train` | `scripts/tok_train.py` | `--max-chars 20M --doc-cap 10000 --vocab-size 16384` | `data/tokenizer/tokenizer.pkl`、`token_bytes.pt (16384,)` |
| `1c. Tokenizer Eval` | `scripts/tok_eval.py` | （无参数） | gpt2/gpt4/ours 压缩率对比、`data/report/tokenizer-evaluation.md` |
| `2a. Base Train - GPT-Nano` | `scripts/base_train.py` | `--depth 2 --max-seq-len 512 --device-batch-size 1 --total-batch-size 32768 --num-iterations 100 --no-compile` | `data/base_checkpoints/d2/{model,optim,meta}_000100.*` |
| `3a. Base Eval` | `scripts/base_eval.py` | `--model-tag d2 --step 100 --eval core,bpb,sample --max-per-task 20 --split-tokens 8192` | `data/base_eval/base_model_000100.csv`、`data/report/*.md` |
| `4a. SFT Mini` | `scripts/chat_sft_mini.py` | `--device-batch-size 4 --num-iterations 200 --max-seq-len 256 --lr 1e-4` | `data/chatsft_checkpoints/d2/model_000200.pt` |
| `6a. Chat CLI` | `scripts/chat_cli.py` | `--source sft --model-tag d2 --step 200 --temperature 0.6 --top-k 50` | 终端交互式对话 |
| `6b. Chat CLI 单次问答` | `scripts/chat_cli.py` | 同上 + `--prompt "你好，请自我介绍一下。"` | 单次回答后退出 |
| `Python: 调试当前文件` | `${file}` | 通用配置 | 调试当前打开的文件（**不追踪**，见 §2.8） |

**6 个脚本之间的流水线关系**（也是各配置的先后依赖）：

```
                ┌────────────────────────────────────────────────────────────────┐
                │  1b tok_train ──► data/tokenizer/{tokenizer.pkl, token_bytes.pt} │   (1c tok_eval 只读)
                └───────────────────────────┬────────────────────────────────────┘
                                            │ vocab_size = 16384
                ┌───────────────────────────▼────────────────────────────────────┐
                │  2a base_train ──► data/base_checkpoints/d2/model_000100.pt     │
                └──────────┬──────────────────────────────────────┬──────────────┘
                           │ 3a base_eval 只读                     │ 4a chat_sft_mini 微调
                           ▼                                      ▼
                data/base_eval/base_model_000100.csv     data/chatsft_checkpoints/d2/model_000200.pt
                                                                  │ 6a/6b chat_cli 加载
                                                                  ▼
                                                              终端对话
```

---

## 1. launch.json 公共环境变量

所有 8 个配置共享同一组 `env`：

```jsonc
"env": {
    "PYTHONPATH": "${workspaceFolder}",                  // 让 nanochat/、tasks/、scripts/ 可被 import
    "NANOCHAT_BASE_DIR": "${workspaceFolder}/data"      // 所有产物根目录（检查点/分词器/数据）
}
```

- `PYTHONPATH=${workspaceFolder}`：debugpy 直接运行脚本文件（不是 `-m` 模块方式），
  因此 `from nanochat.common import ...`、`from tasks.customjson import ...`、
  `from scripts.base_eval import evaluate_core` 都依赖这条路径。
- `NANOCHAT_BASE_DIR=${workspaceFolder}/data`：`nanochat/common.py` 的 `get_base_dir()` 读取它，
  于是 tokenizer / base_checkpoints / chatsft_checkpoints / base_data_climbmix / report 全部落在 `data/` 下。
- 仅 2a 额外有 `TORCH_COMPILE_DISABLE: "1"`（见 §5.3.1，与 `--no-compile` 双保险）。

---

## 2. launch.json 配置逐条解读

### 2.1 `1b. Tokenizer Train` → `scripts/tok_train.py`

```jsonc
"args": [
    "--max-chars", "20000000",   // 训练语料上限 2000 万字符（默认 20 亿的 1/100，CPU 几分钟跑完）
    "--doc-cap", "10000",        // 单文档最多取前 10000 字符
    "--vocab-size", "16384"      // 词表 16384 = 2^14（项目默认 32768 的减半版）
]
```

流程：流式读训练集 → RustBPE 训练 → 存 `tokenizer.pkl` → 无损往返断言 → 生成 `token_bytes.pt`。
详见 §5.1。产物被后续所有配置消费（`get_tokenizer()` 从 `data/tokenizer/` 加载）。

### 2.2 `1c. Tokenizer Eval` → `scripts/tok_eval.py`

无参数。用 7 类文本（英文新闻/韩文/代码/LaTeX/科学文/训练集片段/验证集片段）对比
**gpt2（50257）、gpt4 cl100k_base（~100K）、ours（16384）** 三种分词器的 bytes/token 压缩率。
详见 §5.2。

### 2.3 `2a. Base Train - GPT-Nano` → `scripts/base_train.py`

```jsonc
"args": [
    "--depth", "2",                  // 核心旋钮：2 层 → model_dim=2×64=128 → n_head=128/128=1
    "--max-seq-len", "512",          // 上下文 512 token
    "--device-batch-size", "1",      // 每步 1 条序列（CPU 最小批次）
    "--total-batch-size", "32768",   // 全局批次 32768 token → 需 64 步梯度累积（见 §5.3.5）
    "--num-iterations", "100",       // 只跑 100 步（冒烟测试）
    "--warmup-steps", "10",          // LR 线性预热 10 步
    "--eval-every", "50",            // 每 50 步验证集 BPB
    "--eval-tokens", "65536",        // 验证评估用 65536 token → 128 个 eval step
    "--core-metric-every", "-1",     // 关闭 CORE（省时间，另需下载 eval_bundle）
    "--sample-every", "50",          // 每 50 步采样 7 句提示
    "--save-every", "-1",            // 仅结束时保存检查点
    "--no-compile",                  // Windows 无 MSVC，禁用 torch.compile
    "--run", "dummy"                 // 跳过 wandb 联网日志（DummyWandb）
]
```

深度 2 的模型即 §5.3 的主角：**参数量 ≈ 6.68M，FLOPs/token ≈ 1.59e7**。
（launch 注释写 "~4M params"，实际约 6.7M，原因见 §7.2 偏差说明。）

### 2.4 `3a. Base Eval` → `scripts/base_eval.py`

```jsonc
"args": [
    "--model-tag", "d2",            // 与 2a 的 --run 对应 → base_checkpoints/d2/
    "--step", "100",                // 评估第 100 步检查点
    "--device-type", "cpu",
    "--device-batch-size", "4",     // BPB 评估批大小
    "--eval", "core,bpb,sample",    // 三种评估全开
    "--max-per-task", "20",         // 每个 CORE 任务只取 20 题
    "--split-tokens", "8192"        // BPB 每个 split 只用 8192 token → 4 步（见 §5.4.2）
]
```

### 2.5 `4a. SFT Mini` → `scripts/chat_sft_mini.py`

```jsonc
"args": [
    "--device-batch-size", "4",     // 4 条 256 长序列/步
    "--num-iterations", "200",      // 200 步
    "--max-seq-len", "256",
    "--lr", "1e-4",                 // 所有参数组统一基准 lr（见 §5.5.3）
    "--run", "dummy",
    "--eval-every", "100"           // 本脚本实际未使用该参数（见 §5.5）
]
```

离线 SFT：只用本地 `data/identity_conversations.jsonl`（1000 条身份对话，已实测 1000 行），
不依赖 HuggingFace。加载 2a 产出的 d2 第 100 步模型 → 微调 → 存第 200 步检查点。

### 2.6 `6a. Chat CLI` → `scripts/chat_cli.py`

```jsonc
"args": [
    "--source", "sft",              // chatsft_checkpoints/
    "--model-tag", "d2",
    "--step", "200",                // 4a 的产物
    "--device-type", "cpu",
    "--temperature", "0.6",         // 采样温度
    "--top-k", "50"                 // 每步只在 top-50 里采样
]
```

交互式对话：循环读 stdin → 拼 `<|user_start|>...<|assistant_start|>` → Engine 流式生成。

### 2.7 `6b. Chat CLI - 单次问答`

与 6a 唯一区别：`--prompt "你好，请自我介绍一下。"`（无 `--top-k`，用默认 50）。
执行一次对话后 `break` 退出，不进入交互循环。

### 2.8 `Python: 调试当前文件`（通用，不追踪）

`"program": "${file}"` 直接调试当前打开的任意 Python 文件，无固定入口，
故不参与下面的依赖追踪与形状分析。

---

## 3. 全局依赖树（递归刨根问到底）

下面把 6 个入口脚本 `import` 的项目内模块**递归展开到叶子**。
项目内模块追到底；第三方库（venv 里的包）标注职责后停住——那是"根"，没有再往下的项目代码。

```
.vscode/launch.json
│
├─ 1b → scripts/tok_train.py
│     ├─ nanochat/tokenizer.py        (RustBPETokenizer)
│     │     ├─ rustbpe                 [第三方] Rust 实现的 BPE 训练器
│     │     ├─ tiktoken                [第三方] OpenAI BPE 推理（encode/decode）
│     │     ├─ tokenizers              [第三方] HuggingFace Tokenizer（HuggingFaceTokenizer 用，tok_train 未用）
│     │     └─ (pickle)                序列化 Encoding 到 tokenizer.pkl
│     ├─ nanochat/common.py
│     │     ├─ torch / torch.distributed   [第三方] DDP（未激活，无 RANK 环境变量）
│     │     └─ filelock                [第三方] 下载文件锁
│     ├─ nanochat/dataset.py
│     │     ├─ requests                [第三方] HTTP 下载 shard
│     │     └─ pyarrow.parquet         [第三方] 列式读取 parquet
│     └─ nanochat/report.py
│           ├─ psutil / socket / platform / subprocess   [第三方] 系统信息与 git 命令
│           └─ nanochat/common.py      (get_base_dir / get_dist_info)
│
├─ 1c → scripts/tok_eval.py
│     ├─ nanochat/tokenizer.py         （同上，还用到 from_pretrained("gpt2"/"cl100k_base")）
│     ├─ nanochat/dataset.py           （同上）
│     └─ nanochat/report.py            （同上）
│
├─ 2a → scripts/base_train.py
│     ├─ nanochat/common.py            (preflight_compile_check 在 import gpt 前执行！)
│     ├─ nanochat/gpt.py               ← 模型本体
│     │     ├─ nanochat/common.py      (get_dist_info / COMPUTE_DTYPE)
│     │     ├─ nanochat/optim.py       (MuonAdamW / DistMuonAdamW)
│     │     │     └─ torch.compile     (条件装饰器：TORCH_COMPILE_DISABLE=1 时 no-op)
│     │     └─ nanochat/flash_attention.py
│     │           ├─ kernels [第三方]  FA3 kernel（仅 CUDA SM90；CPU → None → SDPA 回退）
│     │           └─ torch.nn.functional.scaled_dot_product_attention   (SDPA 回退实现)
│     ├─ nanochat/dataloader.py
│     │     ├─ nanochat/common.py, nanochat/dataset.py
│     │     └─ pyarrow.parquet
│     ├─ nanochat/tokenizer.py         (get_tokenizer / get_token_bytes)
│     ├─ nanochat/checkpoint_manager.py
│     │     ├─ nanochat/common.py
│     │     ├─ nanochat/gpt.py         (GPT/GPTConfig → 重建模型)
│     │     └─ nanochat/tokenizer.py
│     ├─ nanochat/loss_eval.py         (evaluate_bpb)
│     ├─ nanochat/engine.py            (采样用)
│     │     └─ nanochat/checkpoint_manager.py（仅 __main__ 测试用）
│     ├─ scripts/base_eval.py          (evaluate_core —— 跨脚本导入！)
│     │     ├─ nanochat/common.py / tokenizer.py / checkpoint_manager.py / dataloader.py / loss_eval.py / engine.py
│     │     ├─ nanochat/core_eval.py
│     │     │     └─ jinja2 [第三方]  prompt 模板渲染
│     │     ├─ transformers [第三方]   仅 --hf-path 时加载 HF 模型（launch 未用）
│     │     └─ yaml / csv / zipfile
│     ├─ nanochat/fp8.py               [仅 --fp8 + CUDA 时 import；CPU launch 是死分支]
│     ├─ nanochat/report.py
│     └─ wandb [第三方]                 (--run dummy → DummyWandb，不联网)
│
├─ 3a → scripts/base_eval.py
│     ├─ nanochat/common.py / tokenizer.py / checkpoint_manager.py
│     ├─ nanochat/core_eval.py         (jinja2)
│     ├─ nanochat/dataloader.py / loss_eval.py / engine.py
│     ├─ nanochat/report.py
│     └─ yaml / csv / zipfile / tempfile / shutil / random
│
├─ 4a → scripts/chat_sft_mini.py
│     ├─ nanochat/common.py            (compute_init / preflight_compile_check)
│     ├─ nanochat/checkpoint_manager.py   (load_model / save_checkpoint)
│     ├─ nanochat/tokenizer.py         (get_tokenizer)
│     ├─ tasks/customjson.py
│     │     └─ tasks/common.py         (Task 基类)
│     └─ (间接) nanochat/gpt.py → optim.py → flash_attention.py
│
├─ 6a/6b → scripts/chat_cli.py
│     ├─ nanochat/common.py            (compute_init / autodetect_device_type)
│     ├─ nanochat/engine.py            (Engine + KVCache)
│     │     └─ nanochat/checkpoint_manager.py
│     └─ nanochat/checkpoint_manager.py   (load_model)
│
└─ 9 → ${file}   （通用配置，不追踪）
```

**第三方依赖职责一句话（刨根到此为止）**：

| 库 | 职责 | 谁在用 |
|---|---|---|
| `rustbpe` | Rust 实现的高性能 BPE **训练** | tok_train |
| `tiktoken` | OpenAI BPE **推理**（encode/decode/batch） | tokenizer（全项目） |
| `tokenizers` | HuggingFace 分词器实现 | tokenizer 的 HuggingFaceTokenizer 类（launch 未用） |
| `pyarrow` | parquet 列式读（row_group 粒度） | dataset / dataloader |
| `requests` | HTTP 流式下载数据分片 | dataset |
| `filelock` | 多进程下载互斥锁 | common |
| `jinja2` | CORE prompt 模板 | core_eval |
| `wandb` | 云端训练日志（`--run dummy` 时不激活） | base_train |
| `psutil`/`socket`/`platform` | 报告卡系统信息 | report |
| `transformers` | 加载 HuggingFace 模型（仅 `--hf-path`） | base_eval（launch 未用） |
| `kernels` | FA3 内核包（仅 CUDA SM90；CPU 返回 None） | flash_attention |

---

## 4. `nanochat/` 包逐模块笔记

> 每个模块：职责 → 关键函数/类 → 关键形状 → 被谁调用。

### 4.1 `nanochat/common.py`（353 行）

**职责**：全项目公共设施——计算精度检测、设备/分布式初始化、日志、下载、峰值算力表、compile 预检。

- `COMPUTE_DTYPE`（模块加载时一次性检测，全局变量）：
  - 有 `NANOCHAT_DTYPE` 环境变量 → 用它（"bfloat16"/"float16"/"float32"）
  - 有 CUDA 且 SM ≥ 8.0 → `bfloat16`
  - 有 CUDA 但 SM < 8.0 → `float32`（fp16 需 GradScaler，未实现，故回退）
  - 无 CUDA（**launch 场景：Windows CPU**）→ **`float32`**
  - 影响：Linear 前向把权重 cast 成它；RoPE 缓存用它；GradScaler 只在 fp16 时启用（launch 下 `scaler=None`）。
- `compute_init(device_type)`：种子 42 → CUDA 时 tf32 matmul → 若 torchrun（有 RANK/LOCAL_RANK/WORLD_SIZE 环境变量）则 `dist.init_process_group`；**launch 下无这些变量 → 单进程，返回 `(False, 0, 0, 1, torch.device("cpu"))`**。
- `autodetect_device_type()`：CUDA > MPS > CPU。
- `get_base_dir()`：读 `NANOCHAT_BASE_DIR`，launch 设为 `data/`。
- `download_file_with_lock()`：filelock + 5 次指数退避（base_eval 下载 eval_bundle 用）。
- `get_peak_flops(name)`：硬编码 GPU BF16 峰值表（CPU 返回 `inf` → MFU 显示 0%）。
- `preflight_compile_check()`：**必须在 import `nanochat.gpt` 之前调用**（optim.py 的 `@torch.compile` 装饰器在 import 时求值）。检查顺序：`--no-compile` 在 argv → `NANOCHAT_NO_COMPILE=1` → 实际试编译一个 2×2 函数。任一命中 → 设 `TORCH_COMPILE_DISABLE=1`。
- `DummyWandb`：`log/finish` 空实现，`--run dummy` 时替代 wandb。

### 4.2 `nanochat/tokenizer.py`（440 行）

**职责**：GPT-4 风格 BPE 分词器。两个实现类 + 项目级便捷函数。

- `SPECIAL_TOKENS`（**9 个**，全部参与词表）：

  ```
  0  <|bos|>              文档分隔（每文档开头 prepend）
  1  <|user_start|>        2  <|user_end|>
  3  <|assistant_start|>    4  <|assistant_end|>
  5  <|python_start|>       6  <|python_end|>      （工具调用）
  7  <|output_start|>       8  <|output_end|>      （工具输出）
  ```

- `SPLIT_PATTERN`：GPT-4 正则（数字分组 `\p{N}{1,2}`，比 GPT-4 的 1,3 省 token——为 32K 小词表调过）。
- `RustBPETokenizer`（**项目默认**）：
  - `train_from_iterator(iter, vocab_size=16384)`：
    `vocab_size_no_special = 16384 − 9 = 16375` → `rustbpe.Tokenizer().train_from_iterator(...)` 训练 16375 个可合并 token（含 256 字节基 token）→ 用合并表构造 `tiktoken.Encoding`，9 个特殊 token 拿编号 **16375..16383**。总词表 16384 ✓（与 checkpoint meta 一致）。
  - `encode(text, prepend/append, num_threads)`：单串 → `encode_ordinary`；列表 → `encode_ordinary_batch`（多线程）。
  - `render_conversation(conversation)` → `(ids, mask)`：SFT 最关键的预处理，mask 规则见 §5.5.1 的图。
  - `render_for_completion`：RL 用（launch 未用）。
  - `save()`：pickle 整个 `tiktoken.Encoding` → `tokenizer.pkl`。
- `HuggingFaceTokenizer`：train + infer 一体（项目注释说实现混乱，默认不用；launch 未用）。
- `get_tokenizer()`：从 `data/tokenizer/tokenizer.pkl` 加载 RustBPETokenizer。
- `get_token_bytes(device)`：加载 `token_bytes.pt`，形状 **`(16384,) int32`**，9 个特殊 token 为 0、其余 16375 个是该 token 的 UTF-8 字节数（实测非零数 = 16375 ✓）。

### 4.3 `nanochat/dataset.py`（179 行）

**职责**：预训练数据集 = ClimbMix-400B 的 parquet 分片；按需从 HF 下载。

- 配置：`BASE_URL = huggingface.co/datasets/karpathy/climbmix-400b-shuffle`，`MAX_SHARD=6542`，
  `data/base_data_climbmix/shard_XXXXX.parquet`。**最后一个分片 shard_06542 固定是验证集**。
  （本机已下载：shard_00000..00007 共 8 个训练分片 + 06542 验证分片。）
- `list_parquet_files()`：扫目录；目录不存在时打警告并回退旧 `base_data/`。
- `parquets_iter_batched(split, start, step)`：逐文件、逐 row_group 流式读 `text` 列 →
  `yield [文档文本, ...]`。train = 除最后外的所有分片；val = 仅最后一个。
- `download_single_file(index)`：requests 流式 1MB 块 → `.tmp` → rename；5 次指数退避。
- `__main__`：`python -m nanochat.dataset -n 170` 多进程下载分片。

### 4.4 `nanochat/dataloader.py`（150 行）

**职责**：预训练数据加载器——**BOS 对齐 + Best-Fit 打包**。

- 核心思想：每行以 `<|bos|>` 开头；打包时先找"能完整放下的最大文档"，找不到就裁剪最短文档填满剩余
  → **100% 利用率（无 padding），代价是 ~35% token 被裁剪**。
- 预分配缓冲区（B=1, T=512 的 launch 场景）：

  | 缓冲区 | 形状 | dtype | 作用 |
  |---|---|---|---|
  | `row_buffer` | `(B, T+1)` = `(1, 513)` | int64 | 逐行构建（x/y 共享，y 是 x 右移 1） |
  | `cpu_buffer` | `(2·B·T)` = `(1024,)` | int64 pinned | CPU 暂存，单次 HtoD |
  | `gpu_buffer` | `(2·B·T)` = `(1024,)` | int64 | 设备侧持久缓冲区（launch 下 device=cpu，是与 cpu_buffer 分开的另一块内存） |
  | `inputs / targets` | `(B, T)` = `(1, 512)` | int64 | 视图切片：x=前 512，y=后 512 |

- `_document_batches(split, resume_state_dict, tokenizer_batch_size=128)`：无限迭代；
  DDP 时 rank 交错取 row_group（launch 无 DDP → rank0 顺序读全部）；记录 `(pq_idx, rg_idx, epoch)` 供断点续训。
- `refill_buffer()`：每批 128 篇文档 → `tokenizer.encode(batch, prepend=bos, num_threads=4)` → 入 `doc_buffer`（≥1000 篇）。
- `tokenizing_distributed_data_loader_with_state_bos_bestfit`：yield `(inputs, targets, state_dict)`。
  不带 state 的简化版 `..._bos_bestfit` 供 eval 用。

### 4.5 `nanochat/gpt.py`（619 行）—— 模型本体（形状重头戏）

**职责**：GPT Transformer。特性：RoPE、QK-Norm、untied wte/lm_head、ReLU² MLP、Post-emb Norm、
无偏置、GQA、滑窗注意力、Smear/Backout/x0/Value-Embedding 等 modded-nanogpt 风格结构。

关键类与形状（通用记号：`B=批, T=序列, C=n_embd, H=n_head, D=head_dim, KV=n_kv_head, V=vocab`）：

- `GPTConfig`：`sequence_len / vocab_size / n_layer / n_head / n_kv_head / n_embd / window_pattern`。
- `norm(x)`：`F.rms_norm(x, (x.size(-1),))`，**无学习参数**。
- `Linear(nn.Linear)`：权重存 **fp32**，前向 `F.linear(x, self.weight.to(x.dtype))` —— 手工混合精度的核心（替代 autocast）。
- `CausalSelfAttention`：
  - 参数：`c_q (C,C)`、`c_k (C, KV·D)`、`c_v (C, KV·D)`、`c_proj (C,C)`；可选 `ve_gate (KV, 12)`。
  - 前向（训练）：
    ```
    x (B,T,C) ─c_q─► (B,T,C) ─view─► q (B,T,H,D)   ← FA3 原生布局，无需 transpose
    x (B,T,C) ─c_k─► (B,T,C) ─view─► k (B,T,KV,D)
    x (B,T,C) ─c_v─► (B,T,C) ─view─► v (B,T,KV,D)
    [有 VE 层] ve (B,T,KV·D) ─view─► (B,T,KV,D)；gate = 3·σ(ve_gate(x[..., :12])) → (B,T,KV)
              v = v + gate.unsqueeze(-1) · ve
    RoPE: q,k ← apply_rotary_emb(x, cos, sin)，cos/sin (1,T,1,D/2)
    QK-Norm: q = norm(q)·1.2,  k = norm(k)·1.2
    flash_attn(q,k,v, causal=True, window_size) → y (B,T,H,D) ─view─► (B,T,C) ─c_proj─► (B,T,C)
    ```
- `MLP`：`c_fc (4C, C)` → ReLU → square → `c_proj (C, 4C)`。激活 `(B,T,4C)`。
- `Block`：Pre-LN 两段残差 `x = x + attn(norm(x)); x = x + mlp(norm(x))`。
- `GPT.__init__`（**在 meta device 上下文中运行**——只造形状不占内存）：
  - `window_sizes`：按 `window_pattern` 平铺，`L=(seq_len,0)`，`S=(ceil(seq_len/4/128)·128, 0)`；**最后一层强制 L**。
  - `wte = Embedding(pad_vocab, C)`（pad 到 64 倍数；16384 已整除 → 不 pad）
  - `lm_head = Linear(C, pad_vocab, bias=False)`（untied，不与 wte 共享）
  - `resid_lambdas (n_layer,)`、`x0_lambdas (n_layer,)`、`smear_gate = Linear(24,1)`、
    `smear_lambda (1,)`、`backout_lambda (1,)`
  - `value_embeds = {str(i): Embedding(pad_vocab, KV·D) | has_ve(i, n_layer)}`；
    `has_ve(i) = (i % 2 == (n_layer-1) % 2)` → **交替层 + 最后一层必有**（d2：只有层 1）。
  - RoPE 缓存：`rotary_seq_len = seq_len·10 = 5120`；`cos/sin (1, 5120, 1, D/2)`，`persistent=False` 不存检查点。
- `init_weights()`（meta 之后真实初始化，全部策略见表 §6.1）。
- `forward(idx, targets=None, kv_cache=None, loss_reduction='mean')` 七步（详细形状流见 §5.3.4 大图）：
  ① RoPE 校验/截取（KV 推理时偏移 `T0=kv_cache.get_pos()`）
  ② `x = wte(idx) → cast → norm`
  ③ Smear：`gate = smear_lambda·σ(smear_gate(x[:,1:,:24]))`，`x[:,1:] += gate·x[:,:-1]`
  ④ 逐层：`x = resid_lambdas[i]·x + x0_lambdas[i]·x0` → VE 注入 → Block
  ⑤ Backout：`x -= backout_lambda · x_backout`（`x_backout` 是第 `n_layer//2` 层之后的残差）
  ⑥ `norm → lm_head → logits (B,T,V)` → 截到 vocab → fp32 → softcap `15·tanh(logits/15)`
  ⑦ 有 targets → `CE(logits.view(BT,V), targets.view(BT), ignore_index=-1)`；否则返回 logits。
- `estimate_flops()`：`6·(matmul 参数数) + Σ 12·H·D·有效窗口长度`（推导见 §5.3.3）。
- `setup_optimizer(...)`：7 组参数分组（见 §5.3.6）。
- `generate(tokens, ...)`：朴素 O(N²) 无 KV Cache 流式生成（调试参考；launch 走 engine.py）。

### 4.6 `nanochat/optim.py`（523 行）

**职责**：**MuonAdamW 混合优化器**——矩阵参数用 Muon（动量 + 正交化），嵌入/标量用 AdamW。

- `_conditional_compile`：`TORCH_COMPILE_DISABLE=1` 时装饰器原样返回函数（**launch 2a 设了这个 → no-op，eager 模式**）。
- `adamw_step_fused(p, grad, exp_avg, exp_avg_sq, ...)`：解耦 weight decay → 动量 → 偏差校正 → 更新。
  状态形状 = 参数形状，fp32。
- `muon_step_fused`（单 GPU 时输入为**同 shape 堆叠**）：
  ```
  stacked_grads / stacked_params / momentum_buffer   : (K, m, n)   K=同shape参数个数
  second_momentum_buffer                            : (K, m, 1)  若 m≥n（tall）
                                                    : (K, 1, n)  若 m<n（wide）
  四步: ① Nesterov动量 ② Polar Express正交化(5次迭代, bf16加速)
       ③ NorMuon方差衰减 ④ cautious weight decay(梯度与参数同号才衰减)
  ```
  - 内部 lr 修正：`lr · max(1, m/n)^0.5`；tall 矩阵 `X.mT@X` 路径、wide 矩阵 `X@X.mT` 路径。
- `MuonAdamW`（单 GPU，launch 用）：`_step_adamw` 逐参数、`_step_muon` 堆叠批量。
- `DistMuonAdamW`：ZeRO-2 风格 3 阶段异步通信（DDP 用；launch 单进程不激活）。

### 4.7 `nanochat/checkpoint_manager.py`（231 行）

**职责**：检查点保存/加载 + 目录约定 + 兼容补丁。

- 文件命名：`model_000100.pt`（模型，仅 rank0）、`optim_000100_rank0.pt`（优化器分片，每 rank）、
  `meta_000100.json`（配置/步数/数据加载器状态/循环状态）。
- `load_model(source, device, phase, model_tag, step)`：
  `source → {"base": base_checkpoints, "sft": chatsft_checkpoints, "rl": chatrl_checkpoints}`；
  `model_tag=None → find_largest_model`（挑 `d<数字>` 最大的）；`step=None → find_last_step`。
- `build_model` 六步：加载 state_dict → CPU 时 bf16→fp32 → 去 `_orig_mod.` 前缀 → 补缺失键 →
  **meta 建壳 → to_empty 分配 → init_weights → load_state_dict(strict, assign)** → 校验 `vocab_size` 一致。
- 兼容补丁：旧检查点补 `window_pattern="L"`、`resid_lambdas=1`、`x0_lambdas=0`。

### 4.8 `nanochat/loss_eval.py`（65 行）

**职责**：**BPB（bits per byte）**——与词表大小无关的损失度量。

```
BPB = Σ_tokens loss(t)·[bytes(t)>0] / (ln2 · Σ_tokens bytes(t))
```
- 逐 token 损失：`model(x, y, loss_reduction='none')` → `loss2d (B,T)`；
- `token_bytes (V,)` 里特殊 token=0 → 不计数；`y=-1`（SFT mask）→ 跳过且避免负索引；
- 分布式时 all_reduce 汇总 nats 与 bytes。

### 4.9 `nanochat/core_eval.py`（294 行）

**职责**：CORE 指标（DCLM 论文）——在上下文中学习（ICL）的多任务基准。

- 三种任务类型：`multiple_choice`（选项共享前缀）、`schema`（选项共享后缀）、`language_modeling`（前缀打分）。
- 流程：jinja2 渲染 prompt → tokenize（prepend BOS）→ 找公共前缀/后缀定位 continuation →
  `stack_sequences` 右侧 BOS 填充成 `(B, T_pad)` → `forward_model` 得到逐位置 loss `(B,T)` 与 argmax `(B,T)` →
  MC/schema 取 continuation 平均 loss 最低的选项；LM 比 argmax 与真实 token 是否全等。
- `evaluate_task`：样本按 `rank, rank+world_size, ...` 分派，all_reduce 汇总正确数。
- 形状细节见 §5.4.1。

### 4.10 `nanochat/engine.py`（393 行）

**职责**：高效推理引擎——KV Cache + 批量采样 + 工具调用状态机。**O(N²) → O(N)**。

- `KVCache`（**FA3 原生布局 `(B,T,H,D)`，无 transpose**）：

  | 成员 | 形状 | 说明 |
  |---|---|---|
  | `k_cache / v_cache` | `(n_layers, B, T_max, KV, D)` | 预分配缓存 |
  | `cache_seqlens` | `(B,) int32` | 每行当前长度（FA3 要求 int32） |
  | `prev_embedding` | `(B,1,C)` | 上一 token 归一化嵌入（供 Smear，decode 时用） |

- `Engine.generate(tokens, num_samples, max_tokens, temperature, top_k, seed)`：
  ① **prefill**：batch=1 的 `KVCache(seq_len=len(tokens))`，一次前向填满 KV → 取最后位置 logits `(1,V)` → `expand(num_samples,-1)`
  ② 建 decode `KVCache(batch=num_samples, seq_len=len(tokens)+max_tokens)` 并 `prefill()` 广播 KV
  ③ decode 循环：每步 `ids (B,1)` → 前向（KV 增量）→ `logits (B,V)` → `sample_next_token` →
     工具调用状态机（`<|python_start|>...<|python_end|>` 之间缓存表达式，遇 end 用 `use_calculator` 计算并强制注入
     `<|output_start|>结果<|output_end|>`）→ yield `(token_column, token_masks)`
  ④ 终止：`<|assistant_end|>` / `<|bos|>` / max_tokens。
- `sample_next_token(logits (B,V))`：温度 0 → argmax；否则 softmax（top_k 先截断）→ multinomial。
- `generate_batch`：非流式收集版（base_train/base_eval 采样用）。

### 4.11 `nanochat/flash_attention.py`（163 行）

**职责**：统一注意力接口，自动 FA3 / SDPA 切换。

- `_load_flash_attention_3()`：仅 **CUDA + SM90（H100）** 才从 `kernels` 包加载 FA3；
  **CPU 上返回 None** → `HAS_FA3=False`、`USE_FA3=False`。
- SDPA 回退 `_sdpa_attention(q,k,v (B,H,T,D))`：完整上下文同长 → `is_causal=True`；
  单 token decode → 裁剪窗口后 `is_causal=False`；滑窗/分块 → 显式 bool mask。
- `flash_attn_func`（训练）：FA3 直传；SDPA 需 `transpose(1,2)` 转 `(B,H,T,D)`，算完转回。
- `flash_attn_with_kvcache`（推理）：FA3 原地更新缓存；SDPA 手动写缓存 `k_cache[:, pos:pos+T] = k` → 取 `[:end_pos]` 全量算。
- **注意**：launch 2a 用默认 `window_pattern="SSSL"`，base_train 会打警告"SDPA 无滑窗加速，建议 --window-pattern L"——功能上 SDPA 回退用显式 mask 实现滑窗，只是慢。

### 4.12 `nanochat/report.py`（430 行）

**职责**：训练报告卡。`get_report().log(section, data)` 把每个阶段的结果写成
`data/report/<slug>.md`（如 `tokenizer-training.md`、`base-model-training---基座模型训练.md`——本机已存在）。
仅 rank 0 记录；`generate()` 汇总成 `report.md`（launch 流程只 `log` 不 `generate`）。

### 4.13 `tasks/common.py` + `tasks/customjson.py`

- `tasks/common.py`：`Task` 基类（切片视图 `start/stop/step`）、`TaskMixture`（混合+打散）、`TaskSequence`、`render_mc`。
- `tasks/customjson.py`：`CustomJSON(Task)`——逐行读 JSONL，每行是 `[{role,content},...]` 消息数组；
  校验角色交替（user→assistant→...）、至少 2 条消息。`get_example(i)` 返回 `{"messages": [...]}` 供
  `tokenizer.render_conversation` 使用。

### 4.14 不在 launch 链路中的模块（仅标注引用关系，不展开）

- `nanochat/fp8.py`：FP8 混合精度训练。仅在 `base_train.py --fp8` **且 CUDA** 时 import（launch 2a 是 CPU → 死分支）。
- `nanochat/execution.py`：仅被 `tasks/humaneval.py` 使用（chat_eval 链路，launch 未涉及）。
- `scripts/chat_sft.py / chat_rl.py / chat_eval.py / chat_web.py`：完整 SFT / RL / 对话评估 / Web 服务，launch 未涉及。

---

## 5. 六大执行链路详解（含全部张量形状）

> 以下形状以 launch.json 实际参数为准。核心推导（见 base_train.py `build_model_meta`）：
> `model_dim = ceil(depth × 64 / 128) × 128 = ceil(2×64/128)×128 = 128`，`n_head = 128/128 = 1`，
> `head_dim = 128`，`n_kv_head = 1`，`vocab = 16384`，`seq_len = 512`。

### 5.1 `1b. tok_train.py` 链路（分词器训练）

```
parquets_iter_batched(split="train")
  └─ 逐 row_group yield [doc, ...]（文本，无形状）
       │ text_iterator(): 每篇截断 doc_cap=10000 字符；累计 20,000,000 字符后停止
       ▼
RustBPETokenizer.train_from_iterator(text_iter, vocab_size=16384)
  ├─ rustbpe.Tokenizer().train_from_iterator(iter, vocab_size_no_special=16375, pattern=SPLIT_PATTERN)
  │     → mergeable_ranks: 16375 个 (bytes → rank)   [含 256 个字节基 token]
  └─ tiktoken.Encoding(mergeable_ranks + special_tokens{9 个 → id 16375..16383})
       → n_vocab = 16375 + 9 = 16384
       ▼
tokenizer.save("data/tokenizer") → tokenizer.pkl（pickle 整个 Encoding，198 KB）
       ▼
无损往返断言：encode(test_text) → decode == 原文
       ▼
token_bytes 计算（为 BPB 用）:
  for token_id in 0..16383: bytes = len(decode([token_id]).encode('utf-8')); 特殊 token → 0
  → torch.tensor(..., dtype=int32) → 形状 (16384,)  → torch.save("data/tokenizer/token_bytes.pt")
```

矩阵/张量形状清单：本链路**没有矩阵运算**，唯一张量是 `token_bytes (16384,) int32`
（实测：非零 16375 个，即 9 个特殊 token 计 0 字节）。

### 5.2 `1c. tok_eval.py` 链路（压缩率对比）

```
3 个分词器:
  gpt2  = RustBPETokenizer.from_pretrained("gpt2")         → vocab 50257,  bos=<|endoftext|>
  gpt4  = RustBPETokenizer.from_pretrained("cl100k_base")  → vocab ~100K
  ours  = get_tokenizer()                                  → vocab 16384
7 类文本 × 3 分词器:
  encoded = encode(text) ; assert decode(encoded) == text   ← 无损往返
  ratio = len(text.encode('utf-8')) / len(encoded)          ← 字节/token 压缩率
输出: 对比表（tokens / ratio / 相对差 %），落盘 data/report/tokenizer-evaluation.md
```

无训练张量；只有 Python 列表统计。

### 5.3 `2a. base_train.py` 链路（基座预训练，重头戏）

#### 5.3.1 启动时序（注意顺序，有坑）

```
import 阶段:
  PYTORCH_ALLOC_CONF=expandable_segments:True        ← 缓解显存碎片（CPU 无感）
  from nanochat.common import preflight_compile_check; preflight_compile_check()
      ← 必须先于 import nanochat.gpt！它看到 argv 里有 --no-compile → 设 TORCH_COMPILE_DISABLE=1
      （launch.json 还直接注入了 TORCH_COMPILE_DISABLE=1，双保险）
  import nanochat.gpt → import nanochat.optim → @_conditional_compile 装饰器在 import 时求值
      → 环境变量已设 → torch.compile 全部变 no-op
  print_banner() → 设备检测 compute_init() → CPU, 单进程, COMPUTE_DTYPE=float32
```

#### 5.3.2 建模三步（meta → to_empty → init_weights）

```
Step 1  build_model_meta(depth=2)              在 torch.device("meta") 上建 GPT —— 只有形状不占内存
        model_dim = ceil(2×64/128)·128 = 128 ; n_head = 1 ; n_kv_head = 1
        GPTConfig(sequence_len=512, vocab=16384, n_layer=2, n_head=1, n_kv_head=1, n_embd=128,
                  window_pattern="SSSL")
Step 2  model.to_empty(device=cpu)             分配真实内存（内容为垃圾值）
Step 3  model.init_weights()                   按 §6.1 的策略初始化所有参数
```

#### 5.3.3 规模核算（launch 2a 实际值）

```
参数量（§6.1 详表）:  6,684,714 ≈ 6.68M
  scaling 参数（矩阵+lm_head）: 2,490,380
FLOPs/token:
  6 × 2,490,380                    = 14,942,280   ← 每个 matmul 参数前向2+反向4
  + 注意力 12·H·D·有效窗口 按层求和:
      层0 window S=128 → 12×1×128×128 =   196,608
      层1 window L=512 → 12×1×128×512 =   786,432
                                      =   983,040
  合计 ≈ 15,925,320 ≈ 1.59e7 FLOPs/token
训练总量: 100 迭代 × 32768 token = 3,276,800 token → 总 FLOPs ≈ 5.2e13
Token:缩放参数比 = 3,276,800 / 2,490,380 ≈ 1.32（Chinchilla ≈ 20，冒烟跑故意小）
```

#### 5.3.4 前向七步 —— d2 全形状数据流图（B=1, T=512）

```
 idx (1,512) int64
   │ ① RoPE 缓存校验: cos/sin (1,5120,1,64) 截取 [:,0:512] → (1,512,1,64)
   ▼ ② 嵌入
 x = wte(idx)                       (1,512) → (1,512,128)
 x = x.to(float32); x = norm(x)     (1,512,128)    [x0 = 保存副本]
   │ ③ Smear (训练路径, T>1)
   │    gate = smear_lambda·σ( smear_gate( x[:,1:,:24] ) )    smear_gate: (1,24) → (1,511,1)
   │    x[:,1:] += gate · x[:,:-1]                    (1,512,128)
   ▼ ④ 逐层 for i in 0,1:
   │    x = resid_lambdas[i]·x + x0_lambdas[i]·x0    (1,512,128)   标量广播
   │    ve = value_embeds["1"](idx).to(fp32)         仅 i==1 → (1,512,128)   [i==0: None]
   │    ┌─ Block ─────────────────────────────────────────────────────┐
   │    │ Attn:  n = norm(x)                                  (1,512,128)
   │    │        q = c_q(n) → (1,512,128) → view (1,512,1,128)
   │    │        k = c_k(n) → view (1,512,1,128)
   │    │        v = c_v(n) → view (1,512,1,128)
   │    │        [i==1] gate = 3·σ(ve_gate(x[...,:12]))   ve_gate (1,12) → (1,512,1)
   │    │               v = v + gate.unsqueeze(-1)·ve     (1,512,1,128)
   │    │        RoPE: x1,x2 = x[...,:64], x[...,64:]     各 (1,512,1,64)
   │    │              y1 = x1·cos + x2·sin ; y2 = -x1·sin + x2·cos → cat → (1,512,1,128)
   │    │        q,k = norm(q)·1.2 , norm(k)·1.2           QK-Norm
   │    │        y = flash_attn(q,k,v, causal=True, window_size=window_sizes[i])
   │    │            SDPA 回退: (1,1,512,128) → y (1,512,1,128)
   │    │        y.view(1,512,128) → c_proj → attn_out    (1,512,128)
   │    │ MLP:   n = norm(x) → c_fc (512,128) → (1,512,512) → ReLU² → (1,512,512)
   │    │        → c_proj (128,512) → (1,512,128)
   │    │ 残差:  x = x + attn_out + mlp_out               (1,512,128)
   │    └─────────────────────────────────────────────────────────────┘
   │    i == n_layer//2 == 1 时: x_backout = x            (1,512,128)   [d2: 最后一层]
   ▼ ⑤ Backout
 x = x - backout_lambda·x_backout                          (1,512,128)
   ▼ ⑥ lm_head
 x = norm(x)                                               (1,512,128)
 logits = lm_head(x)  (16384,128) → (1,512,16384)
 logits = logits[..., :16384].float()                      (1,512,16384) fp32
 logits = 15·tanh(logits/15)                               softcap
   ▼ ⑦ loss
 loss = CE(logits.view(512, 16384), y.view(512), ignore_index=-1)   → 标量 (fp32)
```

窗口配置（d2, pattern="SSSL" 平铺到 2 层，最后一层强制 L）：

```
层0: 'S' → window_sizes[0] = (128, 0)   ← 只看到前 128 token
层1: 'L'（强制）→ window_sizes[1] = (512, 0)  ← 全上下文
```

#### 5.3.5 数据流与梯度累积（launch 2a 数值）

```
total_batch_size = 32768 token
tokens/微步 = device_batch_size(1) × max_seq_len(512) = 512
grad_accum_steps = 32768 / 512 = 64 微步

┌─ 一个迭代 step ────────────────────────────────────────────────┐
│ for micro in 0..63:                                            │
│     loss = model(x, y)          x,y (1,512) → loss 标量        │
│     (loss/64).backward()        ← 梯度在 .grad 上累加          │
│     x, y, state = next(train_loader)   ← 预取下一批            │
│ optimizer.step()                ← 64 个微步梯度平均后更新      │
│ model.zero_grad(set_to_none=True)                              │
└────────────────────────────────────────────────────────────────┘
100 迭代 × 64 微步 × 512 token = 3,276,800 token

数据加载器 (B=1, T=512):
  shard_00000..00007.parquet（train）→ row_group → 128 篇/批 → encode(prepend=<|bos|>)
  → doc_buffer(≥1000) → Best-Fit 打包进 row_buffer (1,513)
  → x = row[:, :-1] (1,512), y = row[:, 1:] (1,512)
  → cpu pinned (1024,) → 单次拷贝 → yield
```

#### 5.3.6 优化器分组（setup_optimizer，launch 2a 实际 LR）

`batch_lr_scale = √(32768/524288) = 0.25`；`dmodel_lr_scale = (128/768)^-0.5 = √6 ≈ 2.449`。

| 参数组 | 参数（d2 实际形状/个数） | 优化器 | 实际 lr（launch） | 其他超参 |
|---|---|---|---|---|
| lm_head | (16384,128) ×1 | AdamW | `0.008×0.25×√6 ≈ 0.0049` | β=(0.8,0.96), wd=0.01 |
| wte | (16384,128) ×1 | AdamW | `0.3×0.25×√6 ≈ 0.1837` | β=(0.8,0.995), wd=0.001 |
| value_embeds | (16384,128) ×1 | AdamW | `0.3×0.25×√6×0.5 ≈ 0.0919` | β=(0.8,0.995), wd=0.01 |
| resid_lambdas | (2,) | AdamW | `0.5×0.25×0.01 = 0.00125` | β=(0.8,0.95), wd=0.05（无 dmodel 缩放） |
| x0_lambdas | (2,) | AdamW | `0.5×0.25 = 0.125` | β=(0.96,0.95), wd=0（无 dmodel 缩放） |
| smear 组 | smear_gate (1,24) + smear_lambda (1,) + backout_lambda (1,) | AdamW | `0.2` 固定 | β=(0.8,0.95) |
| Muon (1,12) | ve_gate ×1 → 堆叠 **(1,1,12)** | Muon | `0.02×0.25 = 0.005`（内再 ×√(m/n)=×1） | 动量 0.95, ns=5, β2=0.9 |
| Muon (128,128) | c_q,c_k,c_v,c_proj ×2 层 = 8 个 → **(8,128,128)** | Muon | 同上（×1） | 同上 |
| Muon (128,512) | mlp.c_proj ×2 → **(2,128,512)** | Muon | 同上（wide, ×1） | 同上 |
| Muon (512,128) | mlp.c_fc ×2 → **(2,512,128)** | Muon | 同上 ×√(512/128)=**×2** | 同上 |

- AdamW 状态：`exp_avg / exp_avg_sq` 与参数同形，fp32。
- Muon 状态：`momentum_buffer (K,m,n)`、`second_momentum_buffer`（tall → `(K,m,1)`；wide → `(K,1,n)`）。
- `weight_decay_scaled = 0.28 × 0.25 × (D_REF/target_tokens) ≈ 2.74`（`D_REF=12×scaling(d12)≈1.17e9`，
  `target_tokens=12×2,490,380≈2.99e7`）→ 训练中再按余弦衰减到 0。

#### 5.3.7 调度器（100 步尺度）

```
lr 乘数:  it<10: (it+1)/10 → 线性预热 ; 10≤it≤35: 1.0 ; it>35: 线性降到 0.05
          (warmdown = round(0.65×100) = 65 → 从第 35 步开始降)

Muon 动量:  it<400: 0.85→0.97 线性 ; 衰减期: 0.97→0.90
Muon wd:    weight_decay_scaled · 0.5·(1+cos(π·it/100)) → 余弦衰减到 0

lrm
1.0 ┤      ┌──────────────────┐
0.5 ┤     ╱                    ╲
0.0 ┴────╱                      ╲───►  0.05
    0   10                     35   100  step
```

#### 5.3.8 循环内钩子（launch 参数触发点）

| 钩子 | 参数 | 触发 | 动作 |
|---|---|---|---|
| 验证 BPB | `--eval-every 50 --eval-tokens 65536` | step 0/50/100 | `evaluate_bpb`，val loader B=1，128 步 |
| CORE | `--core-metric-every -1` | 关闭 | — |
| 采样 | `--sample-every 50` | step 0/50/100（仅主进程） | 7 句提示 × Engine 贪婪生成 16 token |
| 保存 | `--save-every -1` | 仅结束 | `save_checkpoint` → model/optim/meta_000100 |

实测 meta_000100.json：`val_bpb=1.9506`、`smooth_train_loss=6.1369`、`total_training_time≈250s`、
`dataloader_state_dict={pq_idx:0, rg_idx:9, epoch:1}`（即只吃到第 1 个 shard 的第 9 个 row_group）。

### 5.4 `3a. base_eval.py` 链路（基座评估）

#### 5.4.1 CORE（22 个任务 × 20 题）

```
data/eval_bundle/core.yaml → icl_tasks（实测 22 个: hellaswag_zeroshot, jeopardy, bigbench_qa_wikidata,
  arc_easy, arc_challenge, copa, commonsense_qa, piqa, openbook_qa, lambada_openai, hellaswag, winograd,
  winogrande, bigbench_dyck_languages, agi_eval_lsat_ar, bigbench_cs_algorithms, bigbench_operators,
  bigbench_repeat_copy_logic, squad, coqa, boolq, bigbench_language_identification）
每个任务: 打乱(seed 1337) → 取前 20 题 → evaluate_task → evaluate_example:

  fewshot(≤10) + item → jinja2 渲染 → prompts[0..C-1]（C=选项数）
  tokenize(prepend BOS) → 找公共前缀/后缀 → start_idxs / end_idxs
  stack_sequences → input_ids (B=C, T_pad)  int64 [右侧 BOS 填充]
  forward_model: outputs = model(input_ids)      (B, T_pad, 16384)
                 targets = roll(input_ids,-1,1)  (B, T_pad)
                 losses  = CE 逐位置             (B, T_pad)   [最后一列 = nan]
                 predictions = argmax           (B, T_pad)
  MC/schema: answer = argmin_i mean(losses[i, start-1 : end-1])  → 对比 gold
  LM:        predicted[start-1:end-1] == actual[start:end] 全等 → 对/错
结果: accuracy → centered = (acc − 0.01·baseline)/(1 − 0.01·baseline) → core_metric = 均值
落盘: data/base_eval/base_model_000100.csv（本机已存在）
```

#### 5.4.2 BPB（train + val 两个 split）

```
tokens_per_step = device_batch_size(4) × sequence_len(512) × world(1) = 2048
split_tokens 8192 % 2048 == 0 → steps = 8192/2048 = 4 步/split
x,y (4,512) → loss2d (4,512) → 按 token_bytes 加权 → BPB
```

#### 5.4.3 Sample

```
7 句条件采样: engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)  ← 贪婪
8 条无条件:   engine.generate_batch([<|bos|>], num_samples=8, max_tokens=128, temperature=1.0)
```

### 5.5 `4a. chat_sft_mini.py` 链路（离线 SFT）

#### 5.5.1 数据与掩码

```
data/identity_conversations.jsonl（实测 1000 行）
  → CustomJSON 逐行校验（角色交替 user→assistant, ≥2 条消息）
  → train_dataset[i] = {"messages": [...]}
  → tokenizer.render_conversation(conv) → (ids, mask):

     [<|bos|>|<|user_start|> 你好 <|user_end|>|<|assistant_start|> 回复 <|assistant_end|> ...]
     mask   0         0       0         0          0               1       1

     ← 只有 assistant 输出的 token 参与损失（mask=1）
```

#### 5.5.2 打包与张量形状（B=4, T=256）

```
row_capacity = 256 + 1 = 257
data_generator: 每轮预渲染 32 条对话 → 逐条拼进 row，直到塞不下 → 剩余用 <|bos|>(mask=0) 填满
sft_loader:
  batch = tensor(rows)         (4, 257) int64
  x = batch[:, :-1]            (4, 256) int64 → device
  y = batch[:, 1:].clone()     (4, 256) int64 → device
  mask = tensor(mask_rows)     (4, 257) int8  → 用 mask[:, 1:] (4, 256) 对齐 y
  y[mask[:, 1:] == 0] = -1    ← 非 assistant token 不监督
  yield x, y

前向（T=256, 与 5.3.4 相同结构，B=4）:
  x (4,256) → ... → logits (4,256,16384)
  loss = CE(logits.view(1024,16384), y.view(1024), ignore_index=-1)  ← 只对 mask=1 位置求梯度
训练: 200 步，无梯度累积；step%10 打印 EMA loss；无 eval（--eval-every 100 在本脚本中未被使用）
保存: data/chatsft_checkpoints/d2/model_000200.pt + meta_000200.json
```

#### 5.5.3 优化器（与 2a 不同的点）

`model.setup_optimizer(unembedding_lr=1e-4, embedding_lr=1e-4, matrix_lr=1e-4, weight_decay=0.0)`
- `scalar_lr` 未传 → 默认 0.5（x0 组 lr=0.5 较大——本脚本作者的取舍，未覆盖）
- dmodel 缩放 √6 照常 → 嵌入类 lr ≈ 2.45e-4；Muon 组 lr=1e-4（内再按 √(m/n) 修正）
- `weight_decay=0`；模型 `phase="train"` 加载 d2 第 100 步（`find_last_step` 自动找最新）。

### 5.6 `6a/6b. chat_cli.py` 链路（对话推理）

```
load_model("sft", cpu, phase="eval", model_tag="d2", step=200)
  → GPT d2 + tokenizer + meta_000200
特殊 token id: bos, <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>
conversation_tokens = [bos]
循环: 输入 → conversation_tokens += [user_start] + encode(text) + [user_end] + [assistant_start]
  engine.generate(conversation_tokens, num_samples=1, max_tokens=256, temperature=0.6, top_k=50)

  ┌─ ① Prefill（一次前向，B=1）────────────────────────────┐
  │ kv_cache_prefill = KVCache(B=1, seq_len=len(tokens),     │
  │                             2层, 1头, D=128, dtype=fp32) │
  │   k_cache/v_cache  (2, 1, len(tokens), 1, 128)           │
  │ ids = tensor([[tokens]])              (1, T_conv)        │
  │ logits = model(ids, kv_cache)         (1, T_conv, 16384) │
  │   └ 每层: q (1,T_conv,1,128) ; KV 写入缓存 ;             │
  │     注意 Smear 在 prefill(T>1) 用训练同款路径;           │
  │     每层结束 kv_cache.advance? 否——最后一层处理后推进    │
  │ logits[:,-1,:] → (1, 16384) → expand(1,-1)               │
  ├─ ② 广播 ───────────────────────────────────────────────┤
  │ kv_cache_decode = KVCache(B=1, seq_len=T_conv+256, ...)  │
  │   k_cache/v_cache  (2, 1, T_conv+256, 1, 128)            │
  │ prefill() 拷贝 KV 与 prev_embedding                      │
  ├─ ③ Decode 循环（每步）─────────────────────────────────┤
  │ ids (1,1) → forward(kv_cache=decode)                     │
  │   RoPE 偏移 T0=pos ; Smear 用 prev_embedding (1,1,128)   │
  │   q (1,1,1,128); k,v (1,1,1,128) 写入缓存 pos            │
  │   SDPA: k_full (1,pos+1,1,128) → y (1,1,128)             │
  │   → logits (1,1,16384) → sample_next_token(0.6, top_k=50)│
  │   → (1,1) → yield token                                  │
  └─────────────────────────────────────────────────────────┘
  终止: <|assistant_end|> 或 <|bos|>（行完成标记）或 max_tokens=256 → 流式打印
  6b: --prompt 模式 → 单轮后 break
```

---

## 6. 张量/矩阵形状速查表

### 6.1 参数总表（d2，launch 2a 实际配置）

`V=16384, C=128, 4C=512`。init 中 `s = √3/√128 ≈ 0.1531`（均匀分布边界，等 std）。

| 参数 | 形状 | 数量 | 占比 | 初始化 | 优化器组 |
|---|---|---:|---:|---|---|
| `wte.weight` | (16384, 128) | 2,097,152 | 31.4% | N(0, 0.8) | adamw-embedding |
| `lm_head.weight` | (16384, 128) | 2,097,152 | 31.4% | N(0, 0.001) | adamw-unembedding |
| `value_embeds."1".weight` | (16384, 128) | 2,097,152 | 31.4% | U(±s) | adamw-ve |
| 层0: `attn.c_q/c_k/c_v/c_proj` | (128,128) ×4 | 65,536 | 1.0% | U(±s) / c_proj=0 | muon |
| 层0: `mlp.c_fc` | (512,128) | 65,536 | 1.0% | U(±0.4s) | muon |
| 层0: `mlp.c_proj` | (128,512) | 65,536 | 1.0% | 0 | muon |
| 层1: 同上 6 个矩阵 | 同上 | 196,608 | 2.9% | 同上 | muon |
| 层1: `attn.ve_gate` | (1,12) | 12 | ~0 | U(0, 0.02) | muon |
| `resid_lambdas` | (2,) | 2 | ~0 | 1.15→1.05 | adamw |
| `x0_lambdas` | (2,) | 2 | ~0 | 0.20→0.05 | adamw |
| `smear_gate.weight` | (1,24) | 24 | ~0 | U(0, 0.02) | adamw |
| `smear_lambda` | (1,) | 1 | ~0 | 0 | adamw |
| `backout_lambda` | (1,) | 1 | ~0 | 0.2 | adamw |
| **合计** | — | **6,684,714** | 100% | — | — |

> 注意：层 0 无 VE（`has_ve(0,2)=False`），层 1 有（最后一层必开）。三大嵌入各占 31%，合计 94%，
> 这正是 launch 注释说 "~4M" 而实际 ~6.7M 的原因（注释可能按 tied 权重或忽略 VE 估算）。

### 6.2 激活张量总表（前向，d2，训练 B=1/T=512）

| 阶段 | 张量 | 形状 | dtype |
|---|---|---|---|
| 输入 | `idx` / `y` | (1, 512) | int64 |
| 嵌入 | `x = wte(idx)` | (1, 512, 128) | fp32 |
| Smear | `gate` | (1, 511, 1) | fp32 |
| 逐层缩放 | `x0`, `resid·x + x0·x0` | (1, 512, 128) | fp32 |
| VE | `ve`（仅层1） | (1, 512, 128) | fp32 |
| VE 门 | `gate = 3σ(ve_gate(x[...,:12]))` | (1, 512, 1) | fp32 |
| Q/K/V | `q`, `k`, `v` | (1, 512, 1, 128) | fp32 |
| RoPE | `cos`, `sin` | (1, 512, 1, 64) | fp32 |
| 注意力 | `y`（SDPA 内部转 (1,1,512,128)） | (1, 512, 1, 128) | fp32 |
| 输出投影 | `attn_out / mlp_out` | (1, 512, 128) | fp32 |
| MLP 中间 | `c_fc(x)` | (1, 512, 512) | fp32 |
| Backout | `x_backout`（层1后） | (1, 512, 128) | fp32 |
| lm_head | `logits` | (1, 512, 16384) | fp32（softcap 后） |
| 损失 | `loss` | 标量 () | fp32 |

### 6.3 推理张量总表（6a/6b，num_samples=1）

| 阶段 | 张量 | 形状 |
|---|---|---|
| Prefill 输入 | `ids` | (1, T_conv) |
| Prefill KV | `k_cache/v_cache` | (2, 1, T_conv, 1, 128) |
| Prefill logits | `logits` | (1, T_conv, 16384) → 取尾 (1, 16384) |
| Decode 缓存 | `k_cache/v_cache` | (2, 1, T_conv+256, 1, 128) |
| Decode 每步输入 | `ids` | (1, 1) |
| Decode 每步 | `q/k/v` | (1, 1, 1, 128)；`logits` (1, 1, 16384) → 采样 (1, 1) |
| Smear | `prev_embedding` | (1, 1, 128) |

### 6.4 通用公式（任意 depth d、head_dim=128）

```
model_dim  C = ceil(d × 64 / 128) × 128        n_head H = C/128
wte / lm_head / value_embeds: (V, C)           每个 block 矩阵: 4×C² + 2×4C² = 6C²
总参数 ≈ 3·V·C + d·6C²  (V 大时嵌入占主导)
FLOPs/token ≈ 6·(d·6C² + V·C) + Σ_layers 12·H·(C/H)·window_l
```

---

## 7. 附录

### 7.1 实际运行产物（本机 data/ 目录，均已核实存在）

```
data/                                    ← NANOCHAT_BASE_DIR
├─ tokenizer/
│   ├─ tokenizer.pkl                     tiktoken Encoding（198,866 B；vocab 16384）
│   └─ token_bytes.pt                    (16384,) int32，16375 非零
├─ base_data_climbmix/                   shard_00000..00007.parquet + shard_06542.parquet（val）
├─ base_checkpoints/d2/
│   ├─ model_000100.pt                   26.7 MB（≈ 6.68M 参数 × fp32 4B）
│   ├─ optim_000100_rank0.pt             51.9 MB（AdamW 双动量状态为主）
│   └─ meta_000100.json                  val_bpb=1.9506, 训练 250.4s
├─ chatsft_checkpoints/d2/
│   ├─ model_000200.pt
│   └─ meta_000200.json
├─ identity_conversations.jsonl          1000 行（SFT 数据，需手动 curl 下载）
├─ eval_bundle/ + eval_bundle.zip        CORE 评估包（core.yaml 22 任务 + eval_data/）
├─ base_eval/base_model_000100.csv       CORE 结果 CSV
└─ report/*.md                           各阶段报告卡片（tokenizer-training 等 4 个已存在）
```

### 7.2 与 launch.json 注释的偏差说明

| launch 注释说法 | 实际情况 |
|---|---|
| 2a 名称写 "~4M params" | 实测 **6,684,714 ≈ 6.7M**。untied 的 wte+lm_head+VE 三个 (16384,128) 嵌入各 2.1M，注释可能按 tied 估算 |
| 2a 注释 "depth=2, 极小模型" | ✓ 正确；n_embd=128、n_head=1 是代码自动推导的 |
| window_pattern 未在 args 中指定 | 走默认 "SSSL" → 层0 窗口 128、层1 全上下文；SDPA 回退会打效率警告但功能正确 |
| `--eval-every 100`（4a） | chat_sft_mini.py 解析了该参数但训练循环**并未使用**（无 eval 逻辑） |
| 3a `--eval core,bpb,sample` | ✓ 三模式都执行（CORE 22 任务×20 题；BPB 每 split 4 步；7+8 采样） |

### 7.3 与 launch 链路相关的环境/依赖前提

- 4a 依赖 `data/identity_conversations.jsonl`，脚本 assert 存在，否则提示 curl 下载（karpathy-public S3）。
- 3a 的 CORE 依赖 `data/eval_bundle/`，缺失时 `download_file_with_lock` 自动从 S3 下载 zip 并解压。
- 2a/4a 的数据加载依赖 `data/base_data_climbmix/*.parquet`，缺失时需 `python -m nanochat.dataset -n <N>` 下载。
- Windows 无 MSVC：2a 必须 `--no-compile`（且 launch 注入了 `TORCH_COMPILE_DISABLE=1`），
  否则 optim.py 里的 `@torch.compile` 装饰器会在 import 期报错。

### 7.4 关键文件清单（本文档覆盖的源码）

| 文件 | 行数 | 角色 |
|---|---:|---|
| `.vscode/launch.json` | 184 | 本文档入口 |
| `scripts/tok_train.py` | 124 | BPE 训练 |
| `scripts/tok_eval.py` | 248 | 压缩率评估 |
| `scripts/base_train.py` | 908 | 基座预训练 |
| `scripts/base_eval.py` | 382 | 基座评估（CORE/BPB/sample） |
| `scripts/chat_sft_mini.py` | 146 | 离线 SFT |
| `scripts/chat_cli.py` | 113 | 对话 CLI |
| `nanochat/gpt.py` | 619 | 模型本体（形状核心） |
| `nanochat/optim.py` | 523 | MuonAdamW |
| `nanochat/tokenizer.py` | 440 | BPE 分词器 |
| `nanochat/engine.py` | 393 | KV Cache 推理引擎 |
| `nanochat/common.py` | 353 | 公共设施 |
| `nanochat/core_eval.py` | 294 | CORE 指标 |
| `nanochat/checkpoint_manager.py` | 231 | 检查点 |
| `nanochat/dataset.py` | 179 | parquet 数据集 |
| `nanochat/dataloader.py` | 150 | BOS best-fit 加载器 |
| `nanochat/flash_attention.py` | 163 | FA3/SDPA 统一接口 |
| `nanochat/loss_eval.py` | 65 | BPB |
| `nanochat/report.py` | 430 | 报告卡 |
| `tasks/common.py` / `tasks/customjson.py` | 124/65 | SFT 数据任务 |
