# nanochat 笔记：`.vscode/launch.json` 全链路解析

> 本文档对 `.vscode/launch.json` 里每个调试配置实际执行的命令做"刨根问底"式笔记。
> 组织形式：**按六个执行文件分节**。每个执行小节内，把该脚本用到的**每一个方法**在
> **调用处**展开解释；该方法引用的其他文件的方法也顺着解释，一直递归到第三方库边界
> （第三方库用一句话职责作为"根"，见 §3）。
> 共享模块在**首次出现处完整展开**，后续出现处给"紧凑回顾 + 锚点"（约定：`↩ 完整展开见 §x.x.x-N`）。
> 全文记录**张量/矩阵形状**（以 launch.json 实际参数推导出的 **d2 模型** 为具体示例），
> 能图解的地方一律用 ASCII 图（VS Code 内置预览不渲染 mermaid）。

> 事实依据（均已实地核实）：
> - 入口：`.vscode/launch.json`（8 个配置 → 6 个脚本）
> - 本项目代码：`scripts/`（6 个入口）、`nanochat/`（13 个模块）、`tasks/`（2 个）
> - 实际运行产物：`data/tokenizer/`、`data/base_checkpoints/d2/model_000100.pt`、
>   `data/chatsft_checkpoints/d2/model_000200.pt`、`data/eval_bundle/`、`data/report/`
> - 关键元数据：`data/base_checkpoints/d2/meta_000100.json` → `vocab_size=16384, n_layer=2, n_embd=128`
> - 本文档形状按**方案 C** 目标配置记录：launch 2a 增加 `--head-dim 32` → `n_head=4, n_kv_head=4, head_dim=32`
>   （盘上检查点/指标为改动前 1 头产物，重跑后自动更新，见 §6.1）

---

## 目录

- **§0 一页速览**：8 个 launch 配置 → 6 个脚本的映射与流水线关系
- **§1 launch.json 公共环境变量**
- **§2 六个执行文件完整解析**（每个小节：CLI → 调用树 → 逐方法递归展开 → 形状全流程 → 产物）
  - §2.1 `1b → scripts/tok_train.py`（分词器训练）
  - §2.2 `1c → scripts/tok_eval.py`（压缩率对比）
  - §2.3 `2a → scripts/base_train.py`（基座预训练，**重头戏**：GPT/MuonAdamW/加载器/前向大图）
  - §2.4 `3a → scripts/base_eval.py`（CORE/BPB/采样评估）
  - §2.5 `4a → scripts/chat_sft_mini.py`（离线 SFT）
  - §2.6 `6a/6b → scripts/chat_cli.py`（对话推理）
- **§3 全局依赖树与第三方库**
- **§4 模块 × 方法索引**（反向速查：每个方法在哪个执行小节完整展开）
- **§5 张量/矩阵形状速查表**
- **§6 附录**（产物清单 / 与 launch 注释的偏差 / 环境前提 / 文件清单）

---

## 0. 一页速览

| launch 配置 | 入口脚本 | 核心参数 | 主要产物 / 输出 |
|---|---|---|---|
| `1b. Tokenizer Train` | `scripts/tok_train.py` | `--max-chars 20M --doc-cap 10000 --vocab-size 16384` | `data/tokenizer/tokenizer.pkl`、`token_bytes.pt (16384,)` |
| `1c. Tokenizer Eval` | `scripts/tok_eval.py` | （无参数） | gpt2/gpt4/ours 压缩率对比、`data/report/tokenizer-evaluation.md` |
| `2a. Base Train - GPT-Nano` | `scripts/base_train.py` | `--head-dim 32 --depth 2 --max-seq-len 512 --device-batch-size 1 --total-batch-size 32768 --num-iterations 100 --no-compile` | `data/base_checkpoints/d2/{model,optim,meta}_000100.*` |
| `3a. Base Eval` | `scripts/base_eval.py` | `--model-tag d2 --step 100 --eval core,bpb,sample --max-per-task 20 --split-tokens 8192` | `data/base_eval/base_model_000100.csv`、`data/report/*.md` |
| `4a. SFT Mini` | `scripts/chat_sft_mini.py` | `--device-batch-size 4 --num-iterations 200 --max-seq-len 256 --lr 1e-4` | `data/chatsft_checkpoints/d2/model_000200.pt` |
| `6a. Chat CLI` | `scripts/chat_cli.py` | `--source sft --model-tag d2 --step 200 --temperature 0.6 --top-k 50` | 终端交互式对话 |
| `6b. Chat CLI 单次问答` | `scripts/chat_cli.py` | 同上 + `--prompt "你好，请自我介绍一下。"` | 单次回答后退出 |
| `Python: 调试当前文件` | `${file}` | 通用配置 | 调试当前打开的文件（**不追踪**） |

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
- `NANOCHAT_BASE_DIR=${workspaceFolder}/data`：`nanochat/common.py` 的 `get_base_dir()` 读取它
  （完整展开见 §2.1.3-④），于是 tokenizer / base_checkpoints / chatsft_checkpoints /
  base_data_climbmix / report 全部落在 `data/` 下。
- 仅 2a 额外有 `TORCH_COMPILE_DISABLE: "1"`（见 §2.3.2，与 `--no-compile` 双保险）。

---

## 2. 六个执行文件完整解析

### 2.1 `1b. Tokenizer Train` → `scripts/tok_train.py`

#### 2.1.1 定位与 CLI

脚本**没有 `main()` 函数**——从第 1 行到第 124 行是模块级顺序执行的代码。

```jsonc
"args": [
    "--max-chars", "20000000",   // 训练语料上限 2000 万字符（默认 20 亿的 1/100，CPU 几分钟跑完）
    "--doc-cap", "10000",        // 单文档最多取前 10000 字符
    "--vocab-size", "16384"      // 词表 16384 = 2^14（项目默认 32768 的减半版）
]
```

流程一句话：流式读训练集 → RustBPE 训练 → 存 `tokenizer.pkl` → 无损往返断言 →
生成 `token_bytes.pt`（为 BPB 指标用）。产物被后续所有配置消费。

#### 2.1.2 调用树总图

```
scripts/tok_train.py（模块级，自上而下执行）
│
├─ [1] argparse / print 配置                       （标准库 argparse，无形状）
├─ [2] text_iter = text_iterator()                ← 本文件函数，③-1
│      └─ parquets_iter_batched(split="train")    ← nanochat/dataset.py，③-2 完整展开（首次使用）
│           └─ list_parquet_files()               ← nanochat/dataset.py，③-2
│                └─ pyarrow.parquet.ParquetFile   [第三方·根] parquet 列式/row_group 读取
├─ [3] RustBPETokenizer.train_from_iterator(text_iter, 16384)  ← nanochat/tokenizer.py，③-3 完整展开（首次使用）
│      ├─ rustbpe.Tokenizer().train_from_iterator(...)         [第三方·根] Rust 实现 BPE 训练
│      └─ tiktoken.Encoding(...)                               [第三方·根] BPE 推理编码器
├─ [4] get_base_dir()                             ← nanochat/common.py，③-4 完整展开（首次使用）
├─ [5] tokenizer.save(tokenizer_dir)              ← nanochat/tokenizer.py，③-5
├─ [6] tokenizer.encode / tokenizer.decode        ← nanochat/tokenizer.py，③-6 完整展开（无损往返断言）
├─ [7] get_vocab_size / get_special_tokens        ← nanochat/tokenizer.py，③-7
│      └─ tokenizer.decode([token_id]) 循环 → torch.tensor(int32) → torch.save
│           → data/tokenizer/token_bytes.pt (16384,)
└─ [8] get_report().log(section="Tokenizer training", ...)  ← nanochat/report.py，③-8 完整展开（首次使用）
       └─ 写 data/report/tokenizer-training.md
```

#### 2.1.3 逐方法展开

##### ③-1 `text_iterator()`（scripts/tok_train.py L32-51，本文件）

**签名**：无参生成器。**职责**：把 parquet 批次"展平"成单文档流，逐篇截断，累计到 `max_chars` 停止。

```
nchars = 0
for batch in parquets_iter_batched(split="train"):   # 每个 batch = 一个 row_group 的文档列表
    for doc in batch:
        doc_text = doc[:args.doc_cap]                # 每篇最多 10000 字符
        nchars += len(doc_text)
        yield doc_text
        if nchars > args.max_chars:  return          # 累计 > 20,000,000 字符 → 停
```

launch 数值：每篇 ≤ 10,000 字符、总计 > 2,000 万字符后停止。纯 Python 字符串流，无张量。

##### ③-2 `parquets_iter_batched(split, start=0, step=1)` + `list_parquet_files()`（nanochat/dataset.py）—— 完整展开（首次使用）

**模块职责**：预训练数据集 = ClimbMix-400B 的 parquet 分片，按需从 HF 下载。
常量：`BASE_URL = https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main`、
`MAX_SHARD = 6542`、`DATA_DIR = get_base_dir()/base_data_climbmix`。
**最后一个分片 `shard_06542.parquet` 固定是验证集**（本机已下载 8 个训练分片 + 06542）。

- `list_parquet_files(data_dir=None, warn_on_legacy=False)`（L38-76）：
  扫 `DATA_DIR`，排序取所有 `*.parquet`（排除 `*.tmp`）；目录不存在 → 可选警告并回退旧 `base_data/` 目录。
  返回绝对路径列表。
- `parquets_iter_batched(split, start, step)`（L78-109）：

```
assert split in ["train", "val"]
parquet_paths = list_parquet_files()
parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]   # train=除最后, val=仅最后
for filepath in parquet_paths:
    pf = pq.ParquetFile(filepath)                        # 只读元数据，不载入全文件
    for rg_idx in range(start, pf.num_row_groups, step): # start/step 供 DDP 多卡交错（tok_train 用默认 0,1）
        rg = pf.read_row_group(rg_idx)                   # 每次只读一个 row_group
        texts = rg.column('text').to_pylist()            # 'text' 列 → Python 列表，每元素一篇完整文档
        yield texts                                      # yield 一批文档（文本，无形状）
```

- `download_single_file(index)`（L112-171）+ `__main__`（L174-202）：requests 流式 1MB 块 → `.tmp` → rename，
  5 次指数退避；仅供 `python -m nanochat.dataset -n <N>` 多进程（`multiprocessing.Pool`）下载分片使用，
  **launch 链路不调用**（数据已就位）。
- **刨根到此为止**：叶子是 `pyarrow.parquet`（列式存储，row_group 粒度读）与 `requests`（HTTP 下载）。

##### ③-3 `RustBPETokenizer.train_from_iterator(text_iterator, vocab_size)`（nanochat/tokenizer.py）—— 完整展开（首次使用）

**模块职责**：GPT-4 风格 BPE 分词器，两个实现类。本项目默认 **RustBPETokenizer**
（rustbpe 训练 + tiktoken 推理）；`HuggingFaceTokenizer` 仅 `base_eval --hf-path` 用（§2.4.3-② 死分支）。

先看两个模块级常量：

- `SPECIAL_TOKENS`（L21-33，**9 个**，全部参与词表，占最后 9 个 id）：

```
id 16375 <|bos|>             文档分隔（每文档开头 prepend）      id 16376 <|user_start|>    id 16377 <|user_end|>
id 16378 <|assistant_start|>  id 16379 <|assistant_end|>        id 16380 <|python_start|>   id 16381 <|python_end|>
id 16382 <|output_start|>     id 16383 <|output_end|>           （python/output 供工具调用）
```

- `SPLIT_PATTERN`（L40）：GPT-4 正则，但数字分组是 `\p{N}{1,2}`（GPT-4 是 `{1,3}`）——
  为 32K 小词表省 token 空间，作者验证 2 是最优数字分组大小。

`train_from_iterator` 本体（L211-235）——三步：

```
① 特殊 token 不参与训练: vocab_size_no_special = 16384 − 9 = 16375   (assert ≥ 256)
   tokenizer = rustbpe.Tokenizer().train_from_iterator(text_iter, 16375, pattern=SPLIT_PATTERN)
      → 16375 个可合并 token（含 256 个单字节基 token）           [第三方·根 rustbpe]
② pattern = tokenizer.get_pattern()                              # 训练时用的正则（字符串）
   mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}   # dict[bytes → rank]
   tokens_offset = len(mergeable_ranks)                          # = 16375
   special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}   # 9 个 → id 16375..16383
③ enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern,
         mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)   [第三方·根 tiktoken]
   return cls(enc, "<|bos|>")   → __init__（L205-209）: self.enc = enc; self.bos_token_id = encode_special("<|bos|>") = 0
```

**规模核对**：mergeable 16375 + special 9 = **16384 = 2^14** ✓（与 checkpoint meta 的 vocab_size 一致）。
**刨根边界**：`rustbpe`（训练，Rust）、`tiktoken.Encoding`（推理，纯 Python 的 dict + 正则）。

##### ③-4 `get_base_dir()`（nanochat/common.py L81-94）—— 完整展开（首次使用）

```
if os.environ.get("NANOCHAT_BASE_DIR"):   nanochat_dir = 该环境变量值     # launch: D:\project\codex\nanochat\data
else:                                     nanochat_dir = ~/.cache/nanochat  # 默认
os.makedirs(nanochat_dir, exist_ok=True)  # 不存在则创建
return nanochat_dir
```

返回字符串；是全项目所有产物路径的根（tokenizer/、base_checkpoints/、report/ …）。

##### ③-5 `tokenizer.save(tokenizer_dir)`（nanochat/tokenizer.py L321-328）

```
os.makedirs(tokenizer_dir, exist_ok=True)
with open(tokenizer_dir + "/tokenizer.pkl", "wb") as f:
    pickle.dump(self.enc, f)            # 保存的是整个 tiktoken.Encoding 对象
```

实测产物：`data/tokenizer/tokenizer.pkl`，198,866 B。加载路径见 §2.2.3-③ `get_tokenizer`。

##### ③-6 `encode / decode / encode_special / __call__ / id_to_token`（nanochat/tokenizer.py）—— 完整展开

- `encode_special(text)`（L272-276）：`@lru_cache(maxsize=32)` 包装的 `self.enc.encode_single_token(text)` ——
  特殊 token 串 → 单个 id（如 `"<|assistant_end|>"` → 16379）。高频调用，LRU 缓存 32 条。
- `encode(text, prepend=None, append=None, num_threads=8)`（L282-311）：
  - `prepend/append` 可以是**特殊 token 字符串**（走 `encode_special`）或**int id**；
  - `text` 为单串 → `enc.encode_ordinary(text)`（普通 BPE 编码，不含特殊 token），
    前后插入 prepend/append；
  - `text` 为列表 → `enc.encode_ordinary_batch(text, num_threads)`（多线程批量，dataloader 用 4 线程，§2.3.3-09）。
  - 返回 `list[int]`。
- `decode(ids)`（L317-319）：`enc.decode(ids)` → 字符串（**保留特殊 token 文本**）。
- `__call__(*a, **k)`（L313-315）：别名 `encode` —— `tokenizer(prompt, prepend="<|bos|>")` 语法由此而来。
- `id_to_token(id)`（L268-270）：`enc.decode([id])` 单个 id → 字符串。

tok_train 中的用途：`encoded = tokenizer.encode(test_text); assert tokenizer.decode(encoded) == test_text`
—— 中英混排 + emoji 的无损往返断言（失败即训练有问题）。

##### ③-7 `get_vocab_size / get_special_tokens` + token_bytes 生成（L260-266 + 脚本 L91-108）

- `get_vocab_size()`：`self.enc.n_vocab` → **16384**。
- `get_special_tokens()`：`self.enc.special_tokens_set` → 9 个特殊 token 字符串的 set。
- token_bytes 计算（BPB 指标的权重，§2.3.3-12）：

```
token_strings = [decode([id]) for id in range(16384)]        # 先解码全部 16384 个 token
token_bytes  = [ 0  if str in special_set                     # 特殊 token → 0（不计字节）
                    else len(str.encode("utf-8"))  for ... ]  # 普通 token → UTF-8 字节数
torch.tensor(token_bytes, dtype=torch.int32, device='cpu')   → 形状 (16384,) int32
torch.save(→ data/tokenizer/token_bytes.pt)
```

实测：非零 16375 个（即 9 个特殊 token 计 0 字节）✓。

##### ③-8 `get_report().log(section, data)`（nanochat/report.py）—— 完整展开（首次使用）

**模块职责**：训练报告卡。`get_report()`（L460-474）：读 `get_dist_info()`（§2.3.3-01），
**仅 rank 0** 返回 `Report(base_dir/report)`，其余返回 `DummyReport`（log/reset 空操作）。
launch 是单进程 rank0 → 真 `Report`。

- `Report.log(section, data)`（L279-309）：

```
slug = slugify(section)                     # 替换 / \ : * ? " < > | 为 "-"，小写，空格 → "-"
写 data/report/{slug}.md：
    ## {section}                            # 标题
    timestamp: 2026-..-.. ..:..:..          # 当前时间
    逐项遍历 data：跳过 falsy（None/空 dict）；
        str → 原样写；dict → 每键一行 "- k: v"（float 保留 4 位小数，int ≥ 10000 加千分位）
```

tok_train 的调用：`section="Tokenizer training"` → `data/report/tokenizer-training.md`
（本机已存在 ✓）；data 里含 CLI 参数、`train_time`、特殊 token 数、token_bytes 的 min/max/mean/std（仅非零）。
- `Report.generate()`（L311-419）：按 `EXPECTED_FILES` 9 个章节文件 + `header.md` 拼出 `report.md`（含
  Summary 指标表），并复制到项目根 `report.md`。**launch 链路只 `log` 不 `generate`**
  （generate 需手动 `python -m nanochat.report generate`）。
- 其余辅助（`run_command/generate_header/get_git_info/get_gpu_info/get_system_info/estimate_cost/extract/reset`）
  仅被 `generate()/reset()` 使用：git 子进程、psutil/socket/platform 系统信息、GPU 计时价、膨胀统计——
  launch 流程不触发，一句话带过。**刨根边界**：subprocess(git)/socket/platform/psutil。

#### 2.1.4 形状全流程

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
tokenizer.save("data/tokenizer") → tokenizer.pkl（pickle 整个 Encoding，198,866 B）
       ▼
无损往返断言：encode(test_text) → decode == 原文
       ▼
token_bytes 计算（为 BPB 用）:
  for token_id in 0..16383: bytes = len(decode([token_id]).encode('utf-8')); 特殊 token → 0
  → torch.tensor(..., dtype=int32) → 形状 (16384,)  → torch.save("data/tokenizer/token_bytes.pt")
```

本链路**没有矩阵运算**，唯一张量是 `token_bytes (16384,) int32`（实测 16375 非零）。

#### 2.1.5 产物与实测

| 产物 | 说明 | 实测 |
|---|---|---|
| `data/tokenizer/tokenizer.pkl` | pickle 的 tiktoken.Encoding（vocab 16384） | 198,866 B ✓ |
| `data/tokenizer/token_bytes.pt` | (16384,) int32，特殊 token=0 | 16375 非零 ✓ |
| `data/report/tokenizer-training.md` | `get_report().log` 产出 | 存在 ✓ |

---

### 2.2 `1c. Tokenizer Eval` → `scripts/tok_eval.py`

#### 2.2.1 定位与 CLI

无命令行参数（脚本内硬编码）。用 **7 类文本** × **3 种分词器**对比 bytes/token 压缩率：

| 文本 | 内容 | 目的 |
|---|---|---|
| news / korean / code / math / science | 新闻、韩文、Python 代码、LaTeX 论文、科学文 | 5 种典型域 |
| fwe-train / fwe-val | 训练/验证分片第 1 个 row_group 的文档拼接 | 领域内分布 |

| 分词器 | 加载方式 | 词表 |
|---|---|---|
| gpt2 | `RustBPETokenizer.from_pretrained("gpt2")` | 50257 |
| gpt4 | `RustBPETokenizer.from_pretrained("cl100k_base")` | ~100K |
| ours | `get_tokenizer()`（1b 的产物） | 16384 |

#### 2.2.2 调用树总图

```
scripts/tok_eval.py（模块级顺序执行）
│
├─ [1] 7 类文本常量；train_docs/val_docs = next(parquets_iter_batched(...))
│      └─ parquets_iter_batched   ↩ 完整展开见 §2.1.3-②
├─ [2] for tokenizer_name in ["gpt2", "gpt4", "ours"]:
│      ├─ RustBPETokenizer.from_pretrained("gpt2"/"cl100k_base")   ← nanochat/tokenizer.py，③-2 完整展开
│      │    └─ tiktoken.get_encoding(...)                          [第三方·根]
│      └─ get_tokenizer()                                          ← nanochat/tokenizer.py，③-3 完整展开
│           └─ RustBPETokenizer.from_directory → pickle.load(tokenizer.pkl)
│      └─ get_vocab_size()  ↩ §2.1.3-⑦；encode/decode 往返断言 ↩ §2.1.3-⑥
│      └─ ratio = len(text.encode('utf-8')) / len(encoded)         # 字节/token
├─ [3] print_comparison("GPT-2", ...) / ("GPT-4", ...)             ← 本文件函数，③-5
└─ [4] get_report().log("Tokenizer evaluation", [markdown 字符串]) ↩ 完整展开见 §2.1.3-⑧
```

#### 2.2.3 逐方法展开

##### ③-1 文本常量与数据采样（L9-165，本文件）

5 类硬编码文本（`r"""...""".strip()`）；`train_docs = next(parquets_iter_batched(split="train"))` 取
第 1 个 row_group 的文档列表并用 `"\n"` 拼接；val 同理（val 可能为空 → `if val_text` 才加入第 7 类）。
`parquets_iter_batched` ↩ 完整展开见 §2.1.3-②（此处用法：`next()` 只消费第一个 yield）。

##### ③-2 `RustBPETokenizer.from_pretrained(tiktoken_name)`（nanochat/tokenizer.py L246-258）—— 完整展开

```
enc = tiktoken.get_encoding(tiktoken_name)          # [第三方·根] tiktoken 内置词表（gpt2 / cl100k_base）
return cls(enc, "<|endoftext|>")                    # __init__ → bos_token_id = encode_special("<|endoftext|>")
```

要点：tiktoken 里文档分隔符叫 `<|endoftext|>`，nanochat 统一叫 `<|bos|>`（功能相同，都标记新序列开始），
所以这里把 `<|endoftext|>` 当作 BOS 传入。`"gpt2"` → 50257 词表；`"cl100k_base"` → GPT-4 词表（≈100K）。

##### ③-3 `get_tokenizer()`（nanochat/tokenizer.py L478-485）—— 完整展开

```
from nanochat.common import get_base_dir        ↩ §2.1.3-④
tokenizer_dir = base_dir + "/tokenizer"
return RustBPETokenizer.from_directory(tokenizer_dir)
```

`from_directory`（L237-244）：`pickle.load(tokenizer.pkl)` → `cls(enc, "<|bos|>")`。
即加载 **1b 刚训练出的 ours**（vocab 16384）。注释里被注释掉的一行说明也可切回 `HuggingFaceTokenizer.from_directory`。

##### ③-4 三 × 七 编码循环（L174-204，本文件）

对每种分词器 × 每类文本：`encode(text)` → `decode` 无损断言（↩ §2.1.3-⑥）；
`ratio = len(text.encode('utf-8')) / len(encoded)`（**字节/token**，越大压缩越好，常见词 ~3–5）。
结果存 `tokenizer_results[name][text_name] = {bytes, tokens, ratio}`。纯 Python 列表统计，无训练张量。

##### ③-5 `print_comparison(baseline_name, baseline_results, ours_results, all_text)`（L217-265，本文件）—— 完整展开

打印对齐表格（`f"{name:<10} ..."` 定宽）+ ANSI 颜色（绿/红标注孰优）：
`relative_diff = (baseline_tokens − ours_tokens) / baseline_tokens × 100`（正值 = ours 更好）；
ratio 更高者标绿。对 GPT-2、GPT-4 各调一次。

##### ③-6 报告落盘（L272-290，本文件）

手工拼 markdown 字符串（两个对比表）→ `get_report().log(section="Tokenizer evaluation", data=[md])`
↩ 完整展开见 §2.1.3-⑧ → `data/report/tokenizer-evaluation.md`。

#### 2.2.4 形状全流程

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

#### 2.2.5 产物与实测

`data/report/tokenizer-evaluation.md`（本机已存在 ✓）。无其他落盘产物。

### 2.3 `2a. Base Train - GPT-Nano` → `scripts/base_train.py`

#### 2.3.1 定位与 CLI

```jsonc
"args": [
    "--head-dim", "32",              // 方案 C：head_dim=32 → n_head=128/32=4（launch.json 已加，在 --depth 前）
    "--depth", "2",                  // 核心旋钮：2 层 → model_dim=2×64=128
    "--max-seq-len", "512",          // 上下文 512 token
    "--device-batch-size", "1",      // 每步 1 条序列（CPU 最小批次）
    "--total-batch-size", "32768",   // 全局批次 32768 token → 需 64 步梯度累积（§2.3.4）
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

深度 2 的模型是本节主角：**head_dim=32（4 头 × 32 维），参数量 ≈ 6.68M，FLOPs/token ≈ 1.59e7**
（launch 注释写 "~4M params"，实际约 6.7M，原因见 §6.2 偏差说明；launch args **已含**
`--head-dim 32`，若不传则默认 head_dim=128 → 只有 1 头）。
脚本同样**没有 `main()`**，模块级顺序执行，循环体在文件后半部。

#### 2.3.2 启动时序（注意顺序，有坑）+ `preflight_compile_check`（完整展开）

```
import 阶段:
  os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"   ← 缓解显存碎片（CPU 无感）
  from nanochat.common import preflight_compile_check; preflight_compile_check()
      ← 必须先于 import nanochat.gpt！它看到 argv 里有 --no-compile → 设 TORCH_COMPILE_DISABLE=1
      （launch.json 还直接注入了 TORCH_COMPILE_DISABLE=1，双保险）
  import nanochat.gpt → import nanochat.optim → @_conditional_compile 装饰器在 import 时求值
      → 环境变量已设 → torch.compile 全部变 no-op（§2.3.3-08）
  print_banner() → 设备检测 compute_init() → CPU, 单进程, COMPUTE_DTYPE=float32
```

**`preflight_compile_check()`（nanochat/common.py L351-394）—— 完整展开**，按优先级的三段检测：

```
① "--no-compile" in sys.argv          → os.environ["TORCH_COMPILE_DISABLE"]="1"; return   ← launch 命中这里
② os.environ["NANOCHAT_NO_COMPILE"]=="1" → 同上；return
③ 自动探测: @torch.compile(dynamic=False) def f(x): return x.sin().cos()
   f(torch.randn(2,2,device='cpu'))   ← CPU 张量触发 C++/OpenMP Inductor 后端路径
   失败（如 Windows 无 MSVC）→ 设 TORCH_COMPILE_DISABLE=1，继续训练（仅变慢）
```

要点：`@_conditional_compile`（optim.py，§2.3.3-08）在 **import 时**读取该环境变量，
所以必须先设变量再 import `nanochat.gpt`。

`print_banner()`（common.py L159-177）：用 `print0` 打印 nanochat ASCII 横幅（DOS Rebel 字体，
manytools.org 生成）；GBK 控制台 `UnicodeEncodeError` 时优雅降级为一行 `[nanochat]`。

#### 2.3.3 逐方法展开

##### 2.3.3-01 设备检测与计算初始化 —— `autodetect_device_type / get_dist_info / compute_init / COMPUTE_DTYPE / print0 / get_peak_flops / DummyWandb`（nanochat/common.py）—— 完整展开（首次使用）

- `autodetect_device_type()`（L213-226）：`torch.cuda.is_available()` > `mps` > `cpu`，print0 一行。
  launch 下 `--device-type` 未指定 → 自动检测 → **"cpu"**。
- `get_dist_info()`（L195-211）：读 `RANK/LOCAL_RANK/WORLD_SIZE` 三个环境变量；
  缺失（**launch：debugpy 直接跑，无 torchrun**）→ 返回 `(False, 0, 0, 1)`。
  配套 `is_ddp_requested()`（L179-185，三变量都在）与 `is_ddp_initialized()`（L187-193，进程组已建）。
- `compute_init(device_type)`（L228-269）：

```
assert device_type ∈ {cuda, mps, cpu}（cuda/mps 还要确认 torch 后端可用）
torch.manual_seed(42)                          # 全局种子（模型 init_weights 用它）
CUDA: torch.set_float32_matmul_precision("high")   # tf32 matmul（launch CPU 跳过）
is_ddp_requested, rank, local_rank, world = get_dist_info()
DDP 且 CUDA: nccl init_process_group + barrier; 否则 device = torch.device(device_type)
return (is_ddp_requested, rank, local_rank, world, device)
```

  launch 返回值：**`(False, 0, 0, 1, torch.device("cpu"))`** → 单进程、主进程。
- `COMPUTE_DTYPE / COMPUTE_DTYPE_REASON`（L21-41，模块加载时算一次的全局变量）：
  `_detect_compute_dtype()` 优先级：① `NANOCHAT_DTYPE` 环境变量（"bfloat16"/"float16"/"float32"）
  ② 有 CUDA 且 SM ≥ 8.0 → `bfloat16` ③ 有 CUDA 但 SM < 8.0 → `float32`（fp16 需要 GradScaler 未实现）
  ④ 无 CUDA → **`float32`（launch 场景）**。
  影响：`Linear` 前向把权重 cast 成它（§2.3.3-05）、RoPE 缓存 dtype（§2.3.3-05）、
  `scaler` 只在 fp16 时启用（launch 下 `scaler=None`，§2.3.3-08 尾注）。
- `print0(s)`（L143-157）：读 `RANK` 环境变量，非 0 不打印；Windows GBK 控制台 `UnicodeEncodeError` 时
  自动降级为 ASCII。全脚本的 `print0(...)` 输出都经它。
- `get_peak_flops(device_name)`（L291-346）：硬编码 GPU BF16 峰值表（Blackwell/Hopper/Ampere/Ada/AMD/RTX，
  如 h100→989e12）；未知 → 返回 `inf`（MFU 显示 0%）。launch 是 CPU → 分支不进，脚本直接置
  `gpu_peak_flops = float('inf')`。
- `DummyWandb`（L276-283）：`log/finish` 空实现。脚本 L234：`use_dummy_wandb = (args.run=="dummy" or not master)`
  → launch 用 DummyWandb，不联网。
- 附带：`ColoredFormatter / setup_default_logging / logger`（L43-79）在 common.py **import 时**执行，
  INFO 级 ANSI 彩色控制台日志——任何 import common 的脚本都会生效。

##### 2.3.3-02 FA3 状态打印（脚本 L239-253）

`from nanochat.flash_attention import USE_FA3` → CPU 上 `USE_FA3=False` → 打印 `!`×80 警告块
（SDPA 回退 + `window_pattern != "L"` 时提示"滑窗无硬件加速、建议 L"）。
`USE_FA3/HAS_FA3` 的完整展开在 §2.3.3-15。

##### 2.3.3-03 分词器与 token_bytes（脚本 L258-261）

- `tokenizer = get_tokenizer()` ↩ 完整展开见 §2.2.3-③（加载 1b 的 tokenizer.pkl）。
- `token_bytes = get_token_bytes(device=device)`（tokenizer.py L487-498）—— 完整展开：

```
路径 base_dir/tokenizer/token_bytes.pt；assert 存在（提示由 tok_train.py 写出）
with open(..., "rb") as f:  return torch.load(f, map_location=device)   # → (16384,) int32 张量
```

- `vocab_size = tokenizer.get_vocab_size()` ↩ §2.1.3-⑦ → **16384**。

##### 2.3.3-04 建模三步 —— `build_model_meta`（脚本 L290-316，本文件）—— 完整展开

```
Step 1  build_model_meta(depth=2):               # 在 torch.device("meta") 上建 GPT —— 只有形状不占内存
        base_dim  = depth × aspect_ratio = 2×64 = 128
        model_dim = ceil(128 / head_dim(32)) × 32 = 128            # 向上取整到 head_dim 倍数
        num_heads = 128 / 32 = 4                                    # 方案 C: --head-dim 32 → 4 头
        config = GPTConfig(sequence_len=512, vocab_size=16384, n_layer=2, n_head=4,
                           n_kv_head=4, n_embd=128, window_pattern="SSSL")
        with torch.device("meta"):  model_meta = GPT(config)
Step 2  model.to_empty(device=cpu)               # 分配真实内存（内容为垃圾值，比 zeros 快）
Step 3  model.init_weights()                     # §2.3.3-06 完整展开
```

`GPTConfig`（gpt.py L49-61）字段表：`sequence_len=2048 / vocab_size=32768 / n_layer=12 / n_head=6 /
n_kv_head=6 / n_embd=768 / window_pattern="SSSL"`（默认值；本项目只传 7 个关键字段）。
meta 设备是"幽灵设备"：tensor 只有 shape/dtype，不占内存——只用来数参数量、算训练规模。

> **名词区分：`B / T / vocab_size / n_embd`**。
> - `vocab_size`（16384）= 词表里有**多少种** token（`wte` 表的**行数**）；
> - `n_embd`（128）= 每种 token 用**多长**的向量表示（`wte` 表的**列数**，即每个 token 的向量长度 / 模型宽度 d_model）；
> - `T`（512）= 一条序列里有**几个** token（序列长度，`--max-seq-len`，张量的时间维）；
> - `B`（1）= 一批有几条序列（batch 维）。
> `wte.weight` 形状 = `(vocab_size, n_embd)` = `(16384, 128)`，查表就是"按 id 挑一行"；输入
> `x (B, T) = (1, 512)`（512 个整数 token id）→ wte → `(B, T, n_embd) = (1, 512, 128)`。
> 残差流全程宽度恒为 `n_embd`：嵌入输出、每层 Block 进/出、lm_head 输入都是 `(B, T, 128)`；
> 只有 `mlp.c_fc` 临时升到 4×n_embd=512（出来回 128）、`lm_head` 把 128 映射回 vocab_size=16384
> （最后一维变回词表大小）。注意力 `head_dim=32` 是 n_embd 拆成 4 个头后的子向量长度
> （4×32=128），不是独立概念。RoPE 缓存 `(1,5120,1,16)` 的 5120 = 10×512 是给 T 预留的位置数。

##### 2.3.3-05 `GPT.__init__` 模型结构 —— `norm / Linear / has_ve / _compute_window_sizes / _precompute_rotary_embeddings / CausalSelfAttention / MLP / Block`（nanochat/gpt.py）—— 完整展开（首次使用）

先看四个小构件：

- `norm(x)`（L64-68）：`F.rms_norm(x, (x.size(-1),))` —— RMSNorm，**无学习参数**（与 LayerNorm 的区别）。
- `Linear(nn.Linear)`（L75-81）：前向 `F.linear(x, self.weight.to(dtype=x.dtype))` ——
  **手工混合精度核心**：权重存 fp32（优化器精度），matmul 前 cast 到激活 dtype
  （launch 下 COMPUTE_DTYPE=float32 → 实际 no-op）；全部 `bias=False`。
- `has_ve(layer_idx, n_layer)`（L84-87）：`layer_idx % 2 == (n_layer-1) % 2` → 交替层 + **最后一层必开**。
  d2：`has_ve(0,2)=False`、`has_ve(1,2)=True` → 只有层 1 有 Value Embedding。
- `_compute_window_sizes(config)`（L399-432）：

```
long_window  = sequence_len = 512
short_window = ceil(512/4/128)×128 = 128          # S = seq_len/4 向上取整到 128（FA3 tile 对齐）
平铺 window_pattern（d2 共 2 层）→ 最后一层强制 L:
    层0 'S' → (128, 0)   ← 只看到前 128 token
    层1 'L'（强制）→ (512, 0)  ← 全上下文
window_size 是 FA3 的 (left, right) 元组: left=-1=不限, right=0=因果
```

- `_precompute_rotary_embeddings(seq_len, head_dim, base=100000)`（L377-397）：

```
channel_range = arange(0, 32, 2, fp32)               # (16,) 隔步取通道（head_dim=32）
inv_freq = 1 / (100000 ** (channel_range/32))        # (16,) 频率随通道指数衰减
t = arange(seq_len, fp32)                            # (5120,)  rotary_seq_len = 512×10
freqs = torch.outer(t, inv_freq)                     # (5120, 16) = 位置×频率
cos, sin = freqs.cos(), freqs.sin() → 转 COMPUTE_DTYPE → [None,:,None,:]
→ cos/sin (1, 5120, 1, 16)   register_buffer(persistent=False)  ← 不存检查点
```

`GPT.__init__`（L230-284，**在 meta 设备上下文里运行**——只造形状，真实数据在 `init_weights`）：

**真实代码（gpt.py L249-284；`wte`/`h` 的 ModuleDict 解读见下文）：**

```python
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)     # ① 输出投影（untied）
self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))           # ② 每层残差缩放 (2,)
self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))          # ② 每层 x0 混合 (2,)
self.smear_gate    = Linear(24, 1, bias=False)                          # ③ Smear 门控 → 权重 (1,24)
self.smear_lambda  = nn.Parameter(torch.zeros(1))                       # ③ Smear 强度 (1,)
self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))                 # ④ Backout 强度 (1,)
head_dim = config.n_embd // config.n_head                               # 128/4 = 32
kv_dim   = config.n_kv_head * head_dim                                  # 4×32 = 128
self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim)
                                   for i in range(config.n_layer) if has_ve(i, config.n_layer)})  # ⑤ d2: {"1": (16384,128)}
self.rotary_seq_len = config.sequence_len * 10                          # ⑥ 5120 = 512×10
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)   # (1,5120,1,16)
self.register_buffer("cos", cos, persistent=False)                      # ⑦ 缓冲区，不存检查点
self.register_buffer("sin", sin, persistent=False)
```

（`window_sizes` 与 `padded_vocab` 的计算见上文；`wte`/`h` 的详细解读见下方 ModuleDict 小节。）

逐项解读：

**① `self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)` —— 输出投影（语言模型头）**
- 张量：`Linear(in=128, out=16384)`，权重 **(16384, 128)**，参数 2,097,152（占 31.4%）。
- 作用：把残差流 `(B,T,128)` 映射回词表空间 `(B,T,16384)`（每个位置对 16384 个词的打分）→ softcap → CE
  （§2.3.3-15 ⑥⑦）。
- **untied**：与 `wte` **不共享权重**——wte 学"token 的表示"、lm_head 学"预测下一词"；两个矩阵独立优化
  （§2.3.3-08：lm_head AdamW lr≈0.0049 vs wte 0.1837）。
- init `N(0, 0.001)`（§2.3.3-06：输出层用小尺度，避免初始 logits 幅度爆炸）。

**② `resid_lambdas (2,)` / `x0_lambdas (2,)` —— 每层可学习标量（modded-nanogpt 启发）**
- `nn.Parameter(ones/zeros)` 只是**占位假初始化**；真值在 `init_weights`：resid `1.15→1.05`、x0 `0.20→0.05`
  （浅层更强，§2.3.3-06）。
- 作用（GPT.forward ④ 的逐层入口）：`x = resid_lambdas[i]·x + x0_lambdas[i]·x0`
  - `resid_lambdas[i]`：缩放本层残差流（1.0 = 无缩放）——每层一个可学习的"残差增益"；
  - `x0_lambdas[i]`：把**初始归一化嵌入 x0** 按比例混回第 i 层——浅层更依赖原始 token 信息。
- 为什么**分开参数**：优化器独立调度（§2.3.3-08：resid lr=0.00125/β=(0.8,0.95)/wd=0.05；x0 lr=0.125/β=(0.96,0.95)/wd=0）。
- 张量：参数 `(2,)`；前向按层取 `resid_lambdas[i]`（标量）与 `(B,T,128)` 广播乘。

**③ `smear_gate (1,24)` + `smear_lambda (1,)` —— Smear 机制（廉价 bigram 信息）**
- 作用（GPT.forward ③）：把**前一个 token 的嵌入泄漏到当前位置**——
  `gate = smear_lambda·σ(smear_gate(x[:,1:,:24]))` → `(B,T−1,1)`（σ 压缩到 (0,1)，λ 控制强度）；
  `x[:,1:] += gate·x[:,:-1]`（位置 t 混入位置 t−1 的嵌入）。
- `smear_gate` 取输入**前 24 维**做门控（与 ve_gate 取前 12 维同思路：用残差流前几维当"廉价特征"，
  不额外加投影维度）。
- `smear_lambda` init **0 → 初始关闭**，是否启用完全由训练学出。
- 张量：参数 `(1,24)` + `(1,)`；门控激活 `(B,T−1,1)`。

**④ `backout_lambda (1,)` —— Backout 机制**
- 作用（GPT.forward ⑤）：最终归一化前**减去缓存的中层残差**——`x = x − backout_lambda·x_backout`，
  `x_backout` 是第 `n_layer//2` 层（d2: 层1）之后的残差 `(B,T,128)`。
- 目的（作者注释）：去除拼写/词法等**低层表面特征**，让 lm_head 只看到高层语义（"内容净化"）。
- init 0.2 固定（§2.3.3-06）。

**⑤ `value_embeds` —— Value Embedding（ResFormer 风格）**
- 构造：`{str(i): Embedding(padded_vocab, kv_dim) for i if has_ve(i, n_layer)}`；
  `kv_dim = n_kv_head × head_dim = 4×32 = 128`；`has_ve` = 交替层 + **最后一层必开**（上文）。
- d2：`has_ve(0,2)=False`、`has_ve(1,2)=True` → **`{"1": (16384, 128)}`**（参数 2,097,152，31.4%）。
- 作用：按 token id 查表得到"该输出什么"的先验向量；前向（GPT.forward ④）
  `ve = value_embeds[str(i)](idx)` → `(B,T,128)`，传给 Attention 第②步被门控混进 v（上文 forward ②）。
- init `U(±s)`；AdamW-ve 优化器组 lr≈0.0919（§2.3.3-08）。

**⑥ `rotary_seq_len = sequence_len × 10 = 5120` —— RoPE 缓存"过度计算 10 倍"**
- 目的（作者注释）：cos/sin 很小不占内存，**预计算 10 倍序列长**足够；万一超长，GPT.forward ① 会
  **动态扩展**（`T+128` 重算并重新 `register_buffer`）。
- 张量：`freqs (5120, 16)` → cos/sin `(1, 5120, 1, 16)`（batch/head 维为 1 供广播，见上文 `_precompute_rotary_embeddings`）。

**⑦ `register_buffer("cos"/"sin", persistent=False)`**
- `register_buffer`：注册为**缓冲区**——随 `model.to(device)` 迁移、可被 `state_dict` 追踪；
  **`persistent=False` → 不进 state_dict、不存检查点**（cos/sin 可由 `init_weights` 重建，与训练数据无关，
  省 `5120×16×4B×2 ≈ 0.65MB` 检查点体积）。
- 前向会做设备/精度断言：`idx.device == cos.device` 且 `cos.dtype == COMPUTE_DTYPE`（§2.3.3-15 ①）。

**标量参数汇总**：resid `(2,)` + x0 `(2,)` + smear_lambda `(1,)` + backout_lambda `(1,)` +
smear_gate.weight `(1,24)` = **30 个参数** = `num_scaling_params` 的 `scalars` 组（§2.3.3-07）；
优化器里 smear_gate/smear_lambda/backout_lambda 归入 smear 组 lr=0.2（§2.3.3-08）。

**原代码逐行解读：`self.transformer = nn.ModuleDict({...})`（gpt.py L252-255）**：

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),     # ① 词嵌入查表层
    "h":   nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # ② 层堆叠
})
```

- `nn.ModuleDict`：PyTorch 的"子模块字典"容器（key=字符串，value=子模块）。作用：
  ① 把 wte 和 h 统一挂在 `self.transformer` 名下 → state_dict 键自动带 `transformer.wte.*` /
  `transformer.h.*` 前缀（与检查点文件键名一一对应，`load_state_dict` 才能对上，§2.4.3-③）；
  ② 支持属性访问 `self.transformer.wte(idx)` 与下标/迭代 `for i, block in enumerate(self.transformer.h)`。
  注意：普通 dict/list 存子模块不会被 `named_parameters()/state_dict()` 追踪，**必须**用 ModuleDict/ModuleList。
- ① `"wte": nn.Embedding(padded_vocab_size, config.n_embd)` —— 词嵌入查表：
  - 前向 `wte(idx)`：token id `(B, T)` → 嵌入向量 `(B, T, n_embd)`，本质是**查表不是矩阵乘法**
    （所以 `estimate_flops` 不计它的 FLOPs，§2.3.3-07）。
  - 形状：`(padded_vocab_size, n_embd)` → d2 为 **`(16384, 128)`**，参数 2,097,152（占 31.4%）。
  - `padded_vocab_size`：词表向上对齐到 64 的倍数（DDP 与 Tensor Core 效率），16384 已整除 → 不 pad。
  - 权重在 `init_weights` 里 `N(0, 0.8)` 初始化，并 cast 到 COMPUTE_DTYPE（fp32 launch → no-op）。
  - `"wte"` = "word token embedding"，与 `lm_head` **不共享权重**（untied，§2.3.3-06）。
- ② `"h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])` —— 层堆叠：
  - `nn.ModuleList`：子模块的**列表**容器（同理必须用它，普通 list 不注册；支持 `h[i]` 下标访问）。
  - 列表推导按 `layer_idx = 0..1` 生成 2 个 `Block(config, layer_idx)`；`layer_idx` 层索引传进
    Block → CausalSelfAttention，决定该层的三件事：
    ① 滑窗 `window_sizes[layer_idx]`（层0 S=(128,0) / 层1 L=(512,0)，§2.3.3-05 上文的窗口表）；
    ② 是否建 `ve_gate`（`has_ve(layer_idx, 2)`：层0 无 / 层1 有 → (4,12)）；
    ③ 推理时 KV Cache 的层索引（`kv_cache.get_layer_cache(layer_idx)`，§2.3.3-13）。
  - 每个 Block 内部递归创建 `CausalSelfAttention(config, layer_idx) + MLP(config)`（每层 6 个 Linear，见下表）。
  - state_dict 键形如 `transformer.h.0.attn.c_q.weight`、`transformer.h.1.mlp.c_proj.weight`。

> **命名约定说明**：`wte`（word token embedding）与 `h`（hidden layers，transformer 块堆叠）不是
> PyTorch 强制的名字，而是继承自 Karpathy nanoGPT → OpenAI GPT-2 → HuggingFace Transformers 的
> 命名传统（HF GPT-2 的 state_dict 就是 `transformer.wte.*`、`transformer.h.*`、`lm_head.*`；
> nanoGPT 原还有 `wpe` 位置嵌入，nanochat 用 RoPE 取代位置嵌入所以只剩 `wte`）。
> 但在 nanochat 内部它是**硬契约**：state_dict 键名由它决定，检查点保存/加载/兼容补丁
> （§2.4.3-③ 的 `_orig_mod.` 前缀剥离、`_patch_missing_*` 补缺失键、`load_state_dict(strict=True)`）
> 全都依赖这些键——改名即破坏旧检查点兼容。
> nanochat 相对 HF GPT-2 的差异：融合的 `c_attn` 拆成 `c_q/c_k/c_v`；`mlp.c_fc/c_proj` 沿用 HF；
> `value_embeds` 的 key 用层号字符串（如 `"1"`）是项目自创，无外部传统。

d2 实例化结果：

```
self.transformer
├─ .wte : Embedding(16384 → 128)              # 词查表；init N(0, 0.8)
└─ .h   : ModuleList[2]
          ├─ h[0] = Block(层0) → attn(c_q,c_k,c_v,c_proj; ve_gate=无) + mlp(c_fc,c_proj)   窗口 (128,0)
          └─ h[1] = Block(层1) → attn(c_q,c_k,c_v,c_proj; ve_gate=(4,12)) + mlp(c_fc,c_proj) 窗口 (512,0)
```

**`CausalSelfAttention`（gpt.py L103-188）—— 因果自注意力层，详析**

**职责**：Transformer 里唯一的 token 间信息交换通道——每个 token 只能"看"自己及左侧的 token（因果掩码），
按 Q/K 点积相似度加权聚合 V。一条流水线：

```
x (B,T,C) ─Linear投射→ q/k/v ─(可选 VE 门控注入)→ RoPE ─QK-Norm→ Flash Attention ─c_proj→ y (B,T,C)
```

**`__init__`（L110-130）—— 构造与维度推导**：

| 属性/参数 | 代码推导 | d2 值（C=128, H=4, KV=4, D=32） |
|---|---|---|
| `self.head_dim` | `n_embd // n_head`（先断言 `n_embd % n_head == 0`） | 128/4 = **32** |
| `self.c_q` | `Linear(n_embd, n_head × head_dim)` | `Linear(in=128, out=128)`，权重 (128,128) |
| `self.c_k` / `self.c_v` | `Linear(n_embd, n_kv_head × head_dim)`（GQA：KV 头可少于 Q 头） | `Linear(in=128, out=128)`，权重 (128,128) |
| `self.c_proj` | `Linear(n_embd, n_embd)`（输出投影，零初始化） | `Linear(in=128, out=128)`，权重 (128,128) |
| `self.ve_gate_channels` | 常量 12（取输入残差流前 12 维做门控） | 12 |
| `self.ve_gate` | `Linear(12, n_kv_head)`，**仅 `has_ve` 层** | 层1: `Linear(in=12, out=4)`，权重 (4,12)；层0: None |

> **Linear 记法澄清**：`nn.Linear(in_features, out_features)` 的两个参数是**输入/输出特征数**
> （作用在张量**最后一维**），权重矩阵形状 = `(out_features, in_features)`。
> 表里 "权重 (128,128)" 即 `(out, in)`。层的前向 `y = x @ Wᵀ`（无 bias，§2.3.3-05 的 Linear 类），
> 输入张量**不是** 2D 矩阵而是 `(B, T, in)` → 输出 `(B, T, out)`（batch 与序列维原样保留，权重被所有
> token 共享）。在注意力层里 `x` 是 `(1, 512, 128)`，`c_q(x)` 后仍为 `(1, 512, 128)`，靠
> `.view(B,T,H,D)` 把最后的 `out=4×32=128` 维拆成 4 头。

- **GQA（Grouped-Query Attention）约束**：`assert n_kv_head <= n_head and n_head % n_kv_head == 0`
  —— Q 头数必须是 KV 头数的整数倍。KV 头少 → KV Cache 与注意力计算量成比例缩减
  （§2.3.3-13 的 KV Cache 形状里 `KV=n_kv_head`）。d2 是 4:4（无压缩，GQA 未启用效果）。
- **为什么 `c_q/c_k/c_v` 分开写**（而非 GPT-2/HF 的融合 `c_attn`）：形状清晰，且能按 shape 分组进
  Muon 优化器（§2.3.3-08 的 Muon (128,128) 组 = 8 个参数正是这 4 个投影 × 2 层）。
- **`ve_gate` 设计意图**（ResFormer 风格）：取 `x[..., :12]`（残差流前 12 维）过线性层 + `3·sigmoid` →
  门控值 ∈ (0,3)，按头混合 Value Embedding（上文 `value_embeds`）。为什么用前 12 维：作者取残差流
  的"廉价特征"做门控，不额外加投影维度；sigmoid×3 把门控限制在 (0,3)，初始小正值 → 从近中性开始。

**`forward(x, ve, cos_sin, window_size, kv_cache)`（L132-188）—— 前向逐步详解**（d2 具体数值：B=1, T=512, C=128, H=4, KV=4, D=32；模型级整体图见 §2.3.3-15）

```
输入: x (1,512,128) | ve（层1: (1,512,128)；层0: None）| cos/sin 各 (1,512,1,16)
    | window_size（层0: (128,0)；层1: (512,0)）| kv_cache（训练: None）
```

**第 0 步 拆包形状**
```
B, T, C = x.size()          # 1, 512, 128
```
作用：取出 batch/序列/宽度三个数，供后面 `view` 使用。

**第 ① 步 投影出 Q/K/V —— 每个 token 的三种"角色"向量**
```
q = self.c_q(x).view(B, T, n_head, head_dim)        # Linear(128→128): (1,512,128) → (1,512,128) → view → (1,512,4,32)
k = self.c_k(x).view(B, T, n_kv_head, head_dim)     # (1,512,4,32)
v = self.c_v(x).view(B, T, n_kv_head, head_dim)     # (1,512,4,32)
```
作用：把残差流 x 线性变换成三种角色——**q**（这个位置"想问什么"）、**k**（这个位置"是什么"，
当索引用）、**v**（这个位置"携带什么内容"）。`view` 只把最后的 128 维**重新解释**成 4 组 32 维
（FA3 原生布局 `(B,T,H,D)`，无需 transpose），不复制数据。
张量：`(1,512,128) → (1,512,4,32)` ×3。

**第 ② 步 VE 注入 —— 混合 Value Embedding（仅 has_ve 层；训练与推理都有）**
```
ve   = ve.view(B, T, n_kv_head, head_dim)                     # (1,512,4,32)
gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))           # x[...,:12] (1,512,12) → Linear(12→4)
                                                              # → (1,512,4) → sigmoid ∈ (0,1) → ×3 ∈ (0,3)
v    = v + gate.unsqueeze(-1) * ve                            # (1,512,4,1) 广播乘 (1,512,4,32) → (1,512,4,32)
```
作用（ResFormer 风格）：用**输入相关**的门控，把查表得到的 Value Embedding 按头混进 v——给注意力
"这个 token 该输出什么"的强先验。gate ∈ (0,3) 控制混合强度。
张量：`v (1,512,4,32)` 形状不变，数值被修正。

**第 ③ 步 RoPE 旋转位置编码（只作用于 q/k）**
```
q = apply_rotary_emb(q, cos, sin)     # 内部: x1,x2 = q[..., :16], q[..., 16:]   各 (1,512,4,16)
k = apply_rotary_emb(k, cos, sin)     #      y1 = x1·cos + x2·sin; y2 = -x1·sin + x2·cos
                                      #      cat → (1,512,4,32)
```
作用：把"位置"旋进向量——`q_t` 与 `k_t'` 的内积自动携带 `(t−t')` 的相对位置信号
（旋转角度 = 频率 × 位置差，频率随通道指数衰减，见上文 `_precompute_rotary_embeddings`）。v 不参与。
张量：`(1,512,4,32) → (1,512,4,32)`（值被旋转，形状不变）。

**第 ④ 步 QK-Norm + 固定缩放**
```
q, k = norm(q), norm(k)    # RMSNorm 沿最后一维：每个 (…,32) 向量除以其 RMS → 尺度归一
q, k = q * 1.2, k * 1.2    # 温度缩放
```
作用：q/k 的尺度归一化，稳定点积量级；`×1.2` 把注意力"温度"拆成 QK 各一半（内积等效缩放 1.44，
分布更锐利）——模块 docstring 明言比 SDPA 的 softmax_temperature 方案训练更稳。
张量：`(1,512,4,32)` 数值归一化，形状不变。

**第 ⑤ 步 注意力核心计算 —— softmax 加权求和**
```
训练（kv_cache is None）:
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    # FA3: (1,512,4,32) → 内部完成 scores=Q·Kᵀ/√D、因果+滑窗掩码、softmax、加权 v → (1,512,4,32)
    # SDPA 回退: transpose(1,2) → (1,4,512,32) → _sdpa_attention → (1,4,512,32) → transpose 回
推理（kv_cache 非 None）:
y = flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
        cache_seqlens=..., causal=True, window_size=...)    # 完整说明见 §2.3.3-13
最后一层处理完 → kv_cache.advance(T)
```
作用：对每个查询位置 t、每个头：与**左侧窗口内**的所有键算相似度 → softmax 归一成权重 → 加权求和 v。
数学：`y[b,t,h,:] = Σ_{t' ∈ [max(0, t−window), t]} softmax( (q_t·k_t') / √D ) · v_t'`
—— **因果**：只看到 `t' ≤ t`；**滑窗**：只看到最近 window 个（层0: 128，层1: 全 512）。
张量：`(1,512,4,32) → (1,512,4,32)`（H 个头的计算相互独立、可并行）。

**第 ⑥ 步 拼回头 + 输出投影**
```
y = y.contiguous().view(B, T, -1)     # (1,512,4,32) → (1,512,128)（4 头 concat）
y = self.c_proj(y)                    # Linear(128→128): (1,512,128) → (1,512,128)
return y
```
作用：把 4 个头的结果拼回残差流宽度，再过输出投影**混合头间信息**并投影回残差空间。
张量：`(1,512,4,32) → (1,512,128)`。

**设计要点**：
- 第 ⑥ 步的 `view(B,T,-1)` 等价 HF 的 "concat 所有头"：`H×D = C` 恒成立，拼回即残差流宽度。
- 全程只有 ①⑥ 两步含可学习参数（c_q/c_k/c_v/c_proj）；②③④⑤ 无参数
  （has_ve 层的 ve_gate 是额外的小参数，见上文 __init__ 表）。
- 推理时 ⑤ 用 KV Cache：历史 k/v 不再重算，复杂度 O(T²)→O(T)（§2.3.3-13）。

**`MLP`（gpt.py L191-208）—— 前馈网络，详析**

**职责**：逐 token 的"思考"层（**无跨 token 交互**）：升维 → 非线性 → 降维。**4× 扩张 + ReLU² 激活**。

```
x (B,T,128) ─c_fc→ (B,T,512) ─ReLU→square→ (B,T,512) ─c_proj→ (B,T,128)
```

**`__init__`（L195-200）**：
- `self.c_fc = Linear(n_embd, 4·n_embd)`   → d2: Linear(128,512) → 权重 **(512,128)**
- `self.c_proj = Linear(4·n_embd, n_embd)` → d2: Linear(512,128) → 权重 **(128,512)**
- 无 bias（全项目约定）；`c_proj` 零初始化（§2.3.3-06）。

**`forward(x)`（L202-208）—— 前向逐步详解**（输入 x 是 Block 里 `norm(x)` 之后的 `(1,512,128)`）

```
x = self.c_fc(x)          # ① 升维:  Linear(128→512)      (1,512,128) → (1,512,512)
x = F.relu(x).square()    # ② 非线性+门控: ReLU 清零负值 → 平方放大正值   (1,512,512) 数值变换
x = self.c_proj(x)        # ③ 降维:  Linear(512→128)      (1,512,512) → (1,512,128)
return x                  # 输出 (1,512,128)，交还给 Block 做残差加法
```

逐步作用：
- ① `c_fc` 升维 4×：把每个 token 的 128 维向量映射到 512 维"特征空间"——更高维度给非线性变换
  更多容量（可组合出更丰富的特征）。权重 (512,128) 被所有 token 共享。
  张量：`(1,512,128) → (1,512,512)`。
- ② `relu²`：两层线性之间**必须**有非线性，否则 `c_proj(c_fc(x))` 整体退化成单层线性
  （两个 W 可合并成一个）。ReLU 把负值清零（稀疏），平方把正值非线性放大（大值更突出、小值压扁）——
  整体是 GeLU 的廉价近似，等价"门控"（作者注释 "ReLU.square() = cheap gating"），无额外参数、少一次乘法。
  张量：`(1,512,512)` 数值变换，形状不变。
- ③ `c_proj` 降维回 128：把"思考结果"投影回残差流宽度，`c_proj` 零初始化（§2.3.3-06）→ 初始零贡献。
  张量：`(1,512,512) → (1,512,128)`。

**为什么 4× 扩张**：GPT 系列标准（中间宽度 = 4·n_embd，与模型宽度成比例）；d2 中间宽度 = 512。
**MLP 有没有参数？**——**有**：`c_fc.weight (512,128)` + `c_proj.weight (128,512)` = 每层 **131,072** 个
（2 层共 262,144，占 transformer_matrices 393,264 的大头）；没有参数的是 `norm`（RMSNorm）、
ReLU² 激活与 bias（全项目 `bias=False`）。两者都进 Muon 优化器（§2.3.3-08：c_fc→(512,128) 组、
c_proj→(128,512) 组），初始化 c_fc `U(±0.4s)`、c_proj 全 0（§2.3.3-06）。
**它们反向传播吗？**——**参与**：c_fc/c_proj 都是 `nn.Parameter`，`loss.backward()` 按链式法则
`∂L/∂c_proj.weight = (∂L/∂y)ᵀ@x`、`∂L/∂c_fc.weight` 再穿过 relu²（导数 `2·relu(x)·1[x>0]`）回传，
梯度与参数同形（(512,128)/(128,512)），`optimizer.step()` 更新。注意**无参数 ≠ 无梯度流过**：
relu² 和 `norm` 没有可学习参数，但梯度照样穿过它们传给 c_fc；零初始化的 c_proj 梯度非零，
第一步就会推离 0（§2.3.3-11 的 autograd 说明）。
**补充**：MLP **无跨 token 交互**——每个 token 独立走同一套 `c_fc/relu²/c_proj`（权重共享）；
token 间的信息交换只发生在 Attention（§2.3.3-05 上文）。

**`Block`（gpt.py L211-227）—— 组合单元**

```
x = x + attn(norm(x), ve, cos_sin, window_size, kv_cache)   # Pre-LN: 先 norm 再子层，残差直连
x = x + mlp(norm(x))
return x
```

- 每个 `Block` = `CausalSelfAttention` + `MLP`，两段各自 **Pre-LN**（`norm` 是上文无参数的 RMSNorm）。
- Pre-LN 的意义：norm 放在子层**之前** → 残差流保持单位增益，深层网络稳定（相比 Post-LN 更难训）。
- d2 共 2 个 Block；`layer_idx` 决定该层滑窗与 VE 配置（上文 ModuleDict 解读的 ①②③）。

- `get_device()`（L434-436）：从 `wte.weight.device` 推断模型所在设备（多处使用）。

##### 2.3.3-06 `init_weights()` 初始化策略（gpt.pyL286-375） —— 完整展开 + 三个死分支简注

`@torch.no_grad()`（关闭梯度追踪的装饰器：函数内所有算子不建计算图、不存反向中间量，省内存且加速；
`torch.no_grad()` 本身同时是上下文管理器，可写 `with torch.no_grad():`。**实测去掉它的后果**：在训练前
裸调用不会报错（init 在叶子参数上 in-place，尚未建图），训练也能正常跑；但**训练后再调用会静默
重随机化权重**——训练好的参数被悄悄重置且无任何警告，因此它是"防呆"护栏，应保留），集中初始化全模型
（bound = √3·std 使均匀分布与正态同标准差；d2 的 `s = √3/√128 ≈ 0.1531`）：

| 参数 | d2 形状 | 初始化 |
|---|---|---|
| `wte.weight` | (16384,128) | `N(0, 0.8)` |
| `lm_head.weight` | (16384,128) | `N(0, 0.001)`（untied） |
| `attn.c_q / c_k / c_v` | (128,128) ×3/层 | `U(±s)`（均匀避免离群值） |
| `attn.c_proj` | (128,128) | **0**（输出投影零初始化） |
| `mlp.c_fc` | (512,128) | `U(±0.4s)`（MLP 降幅） |
| `mlp.c_proj` | (128,512) | **0** |
| `resid_lambdas[i]` | (2,) | `1.15 − 0.10·i/(n_layer−1)` → 层0=1.15, 层1=1.05 |
| `x0_lambdas[i]` | (2,) | `0.20 − 0.15·i/(n_layer−1)` → 层0=0.20, 层1=0.05 |
| `smear_lambda` | (1,) | 0（初始关闭） |
| `backout_lambda` | (1,) | 0.2 |
| `smear_gate.weight` | (1,24) | `U(0, 0.02)`（小正值 → 门控近中性） |
| `value_embeds."1".weight` | (16384,128) | `U(±s)`（类同 c_v） |
| `attn.ve_gate.weight`（层1） | (4,12) | `U(0, 0.02)` |

**四个初始化函数笔记（`torch.nn.init.*`，全部**就地 in-place** 修改张量并返回它，在 `@torch.no_grad()` 下运行）**：

| 函数签名 | 数学 | d2 用途 |
|---|---|---|
| `normal_(tensor, mean=0.0, std=1.0)` | 高斯 `N(mean, std²)`（内部用两次 U(0,1) 做 Box-Muller 变换生成） | `wte`：`N(0, 0.8)`（大 std → 嵌入值分散，nanoGPT 传统）；`lm_head`：`N(0, 0.001)`（输出层小尺度 → 初始 logits 接近 0，防 CE/softmax 饱和） |
| `uniform_(tensor, a=0.0, b=1.0)` | 均匀 `U(a,b)`，均值 (a+b)/2、方差 (b−a)²/12 | `c_q/c_k/c_v/ve`：`U(−s, s)`（s=√3/√128≈0.1531）——均值 0、方差 (2s)²/12 = s²/3 = **1/128，与 `N(0, 1/128)` 同标准差**；作者用均匀是为了**避免正态的离群值**（"weights use Uniform to avoid outliers"）；`c_fc`：`U(±0.4s)`（×0.4 降幅防激活爆炸）；`smear_gate/ve_gate`：`U(0, 0.02)`（小正值 → 门控从近中性开始） |
| `zeros_(tensor)` | 全 0 | `attn.c_proj` / `mlp.c_proj`（**输出投影零初始化** → 子层初始输出为 0 → 残差流初始恒等，训练早期像"浅层模型"，稳定）；`smear_lambda`（初始关闭 Smear） |
| `constant_(tensor, val)` | 全填常数 val | `backout_lambda = 0.2` |

补充三点：
- **`Parameter(ones/zeros)` vs 真初始化**：`GPT.__init__` 里的 `nn.Parameter(torch.ones/zeros(...))` 只是 meta 期的**占位假初始化**；真值一律在 `init_weights` 里用上面四个函数 + 直接赋值覆盖（§2.3.3-05 ②）。
- **直接赋值**：`resid_lambdas/x0_lambdas` 用 `param.data[i] = 1.15 − 0.10·i/...`（`.data` 绕开 autograd），不是 nn.init 函数。
- **bound = √3·std 的由来**：均匀 `U(−s, s)` 的方差 = (2s)²/12 = s²/3，要让它与 `N(0, σ²)` 同标准差 → 取 `s = √3·σ`；文档里 `s = √3·(1/√128)`。
- 时序：就地初始化发生在 `to_empty` 之后、任何 forward 之前（§2.3.3-04 的三步），此时参数是叶子且未建图——所以去掉 `no_grad` 也不会报错（§2.3.3-06 上文实测）。

收尾：重算 `cos/sin` 赋给 `self.cos/self.sin`（meta 期是假张量）；若 `COMPUTE_DTYPE != fp16`，
`wte` 与 `value_embeds` **cast 到 COMPUTE_DTYPE 省显存**（launch fp32 → no-op）。

三个死分支（launch 下不执行，各一句）：

- **resume**（L343-348）：`--resume-from-step -1` → 不执行。否则 `load_checkpoint(checkpoint_dir, step, device,
  load_optimizer=True, rank)` → `load_state_dict(strict, assign)`。`load_checkpoint` 完整展开见 §2.4.3-③。
- **fp8**（L356-382）：`--fp8` 未设；且 CPU 会直接警告忽略。`Float8Linear/convert_to_float8_training`
  仅 CUDA SM90 有意义。`disable_fp8(model)` 上下文管理器（L386-440）：把 Float8Linear 临时换成 Linear
  再恢复（评估用 BF16）；launch 下无 Float8 模块 → `yield` 空转。
- **compile**（L446-454）：`TORCH_COMPILE_DISABLE=1` → `use_compile=False` → 模型保持 eager，
  `orig_model` 与 `model` 是同一对象（评估/采样/保存都用 orig_model）。

##### 2.3.3-07 规模核算与缩放律 —— `num_scaling_params / estimate_flops / get_scaling_params`—— 完整展开

**这三个方法各回答一个问题（分工图）**：

```
num_scaling_params()          ← 问题①「模型有多大？参数都长在哪？」
  │                              把参数按优化器分组数清楚（wte/ve/lm_head/transformer/scalars）
  │                              → 打印给用户看 + 供缩放定律实验选"哪组参数最干净"
  ├──► estimate_flops()       ← 问题②「训练 1 个 token 要多少算力？」
  │                              6×matmul参数 + 注意力 12·H·D·窗口 → 总算力/MFU/ETA
  │
  └──► get_scaling_params()   ← 问题③「该训多少 token？lr/批次/权重衰减怎么调？」
        （脚本内函数）            取 transformer_matrices + lm_head 这组"缩放参数"
                                   → target_tokens = 12 × 缩放参数（最优训练量）
                                   → D_REF（d12 参考模型）→ 批次/学习率/权重衰减的 muP 式外推
```

三者关系：`num_scaling_params` 是基础（分组计数），`estimate_flops` 与 `get_scaling_params` 都从它的分组里
取数（前者用"总参数减嵌入"估算算力，后者用"矩阵+lm_head"外推训练规模）。

- `num_scaling_params()`（gpt.py L469-501）：按优化器分组计数（d2 实测）：

| 组 | wte | value_embeds | lm_head | transformer_matrices | scalars | **total** |
|---|---:|---:|---:|---:|---:|---:|
| 数量 | 2,097,152 | 2,097,152 | 2,097,152 | 393,264 | 30 | **6,684,750** |

  （transformer_matrices = 每层 6 个矩阵（c_q/c_k/c_v/c_proj 各 16,384 + c_fc 65,536 + c_proj 65,536
  = 196,608）×2 层 = 393,216，再加层1 的 ve_gate (4,12)=48 → 393,264；
  scalars = 2+2+24+1+1 = 30。assert total == 全部参数和 ✓）
- `estimate_flops()`（gpt.py L438-467）—— **FLOPs/token（前向+反向）**：

```
matmul 参数（排除 wte/ve/标量，只算 lm_head + transformer 矩阵）= 2,490,416
每个参数: 前向 2 FLOPs（乘+加）+ 反向 4 FLOPs = 6
6 × 2,490,416                    = 14,942,496
注意力 QK 内积 12·H·D·有效窗口 按层求和（H=4, D=32，H×D=128）:
    层0 window S=128 → 12×4×32×128 =   196,608
    层1 window L=512 → 12×4×32×512 =   786,432
                                      =   983,040
合计 ≈ 15,925,536 ≈ 1.59e7 FLOPs/token
```

  （与 Chinchilla 公式差 ~1%：不计嵌入查表与 softmax 的 exp/sum/div。）
- `get_scaling_params(m)`（脚本 L476-485，本文件）：`transformer_matrices + lm_head` =
  **2,490,416**（作者实测这是"最干净的缩放定律曲线"组合）。
- 缩放律四步（脚本 L487-536）：
  - `target_tokens = 12 × 2,490,416 = 29,884,992`（--target-param-data-ratio 12；本次被 --num-iterations 100 覆盖，只用它算 wd 缩放）
  - 参考模型 `d12_ref = build_model_meta(12)` → `D_REF = 12 × scaling(d12) ≈ 1.17e9`；`B_REF = 2^19 = 524,288`
  - `total_batch_size = args.total_batch_size = 32768`（用户显式给，跳过 Power-Lines B∝D^0.383 自动推导）
  - `batch_lr_scale = √(32768/524288) = 0.25`（η ∝ √(B/B_ref)）
  - `weight_decay_scaled = 0.28 × 0.25 × (D_REF/target_tokens) ≈ 2.74`（T_epoch 框架 λ=λ_ref·√(B/B_ref)·(D_ref/D)）

##### 2.3.3-08 优化器 —— `setup_optimizer`（gpt.py）→ `MuonAdamW`（nanochat/optim.py）—— 完整展开（首次使用）

**`setup_optimizer(unembedding_lr, embedding_lr, matrix_lr, weight_decay, scalar_lr)`（gpt.py L503-562）**：

```
model_dim = 128;  dmodel_lr_scale = (128/768)^-0.5 = √6 ≈ 2.4495   # AdamW 组按 ∝1/√dmodel 缩放
matrix_params = 全部 transformer.h 参数；value_embeds/wte/lm_head 各自参数
resid_params = [resid_lambdas];  x0_params = [x0_lambdas]
smear_params = [smear_gate.weight, smear_lambda, backout_lambda]
assert 分组无遗漏
param_groups = [ 6 个 AdamW 组（见下表）] + [ 按 shape 排序的 Muon 组（同 shape 堆叠） ]
Factory = DistMuonAdamW if ddp else MuonAdamW     # launch 单进程 → MuonAdamW
optimizer = Factory(param_groups);  每组记 initial_lr = lr（供调度器 §2.3.3-10 乘乘数）
```

launch 2a 实测 10 组（`batch_lr_scale=0.25`、`dmodel_lr_scale=√6` 均已折算）：

| 参数组 | 参数（d2 实际形状/个数） | 优化器 | 实际 lr（launch） | 其他超参 |
|---|---|---|---|---|
| lm_head | (16384,128) ×1 | AdamW | `0.008×0.25×√6 ≈ 0.0049` | β=(0.8,0.96), wd=0.01 |
| wte | (16384,128) ×1 | AdamW | `0.3×0.25×√6 ≈ 0.1837` | β=(0.8,0.995), wd=0.001 |
| value_embeds | (16384,128) ×1 | AdamW | `0.3×0.25×√6×0.5 ≈ 0.0919` | β=(0.8,0.995), wd=0.01 |
| resid_lambdas | (2,) | AdamW | `0.5×0.25×0.01 = 0.00125` | β=(0.8,0.95), wd=0.05（无 dmodel 缩放） |
| x0_lambdas | (2,) | AdamW | `0.5×0.25 = 0.125` | β=(0.96,0.95), wd=0（无 dmodel 缩放） |
| smear 组 | smear_gate (1,24) + smear_lambda (1,) + backout_lambda (1,) | AdamW | `0.2` 固定 | β=(0.8,0.95) |
| Muon (4,12) | ve_gate ×1 → 堆叠 **(1,4,12)** | Muon | `0.02×0.25 = 0.005`（内再 ×√(m/n)：4/12<1 → ×1） | 动量 0.95, ns=5, β2=0.9 |
| Muon (128,128) | c_q,c_k,c_v,c_proj ×2 层 = 8 个 → **(8,128,128)** | Muon | 同上（×1） | 同上 |
| Muon (128,512) | mlp.c_proj ×2 → **(2,128,512)** | Muon | 同上（wide, ×1） | 同上 |
| Muon (512,128) | mlp.c_fc ×2 → **(2,512,128)** | Muon | 同上 ×√(512/128)=**×2 → 0.01** | 同上 |

（Muon 组 `weight_decay = weight_decay_scaled ≈ 2.74`，训练中再按余弦衰减，§2.3.3-10。）

**`nanochat/optim.py` 全体 —— 完整展开**：

- `_conditional_compile(fn, dynamic, fullgraph)`（L32-46）：装饰器。`TORCH_COMPILE_DISABLE` 已设 → 原样返回函数；
  否则 `torch.compile(fn, dynamic, fullgraph)`。launch 2a 下 no-op（eager）。
- `adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)`（L55-87）——
  经典 AdamW 融合内核（超参都是 **0-D CPU 张量**：值变形状不变 → torch.compile 不重编译）：

```
p *= (1 − lr·wd)                         # 解耦权重衰减（先衰减参数）
m = β₁·m + (1−β₁)·grad                  # exp_avg.lerp_(grad, 1−β₁)
v = β₂·v + (1−β₂)·grad²
偏差校正 m̂=m/(1−β₁ᵗ), v̂=v/(1−β₂ᵗ)      # 修正初期矩估计偏低
p -= lr · m̂ / (√v̂ + ε)
```

  状态形状 = 参数形状，fp32（`exp_avg/exp_avg_sq = zeros_like(p)`）。
- `muon_step_fused(...)`（L142-206）—— Muon 四步（输入为**同 shape 堆叠** (K,m,n)）：

```
① Nesterov 动量: m = μ·m + (1−μ)·∇L ;  g = ∇L + μ·m
② Polar Express 正交化（5 次迭代，ns_steps=5；COMPUTE_DTYPE=bf16 时转 bf16 加速）:
   X = g / (‖g‖·1.01 + 1e-6)
   迭代 X = a·X + X·(b·A + c·A²)      tall (m>n): A = X.mT @ X   ← (n,n)
   迭代 X = a·X + (b·A + c·A²)·X      wide (m≤n): A = X @ X.mT   ← (m,m)
   （系数 polar_express_coeffs 5 组；产出近似正交矩阵 US'V^T，S'~U(0.5,1.5)，对性能无损）
③ NorMuon 方差衰减: v_mean = mean(g², dim=red_dim);  second_momentum_buffer lerp(β2)
   step_size = clamp(second_momentum,1e-10).rsqrt()  → 按行/列自适应尺度归一化
④ cautious weight decay + 更新: mask = (g·p ≥ 0)（梯度与参数同号才衰减）
   p -= lr·g + lr·wd·p·mask
```

  状态形状：`momentum_buffer (K,m,n)`；`second_momentum_buffer` tall → **(K,m,1)**、wide → **(K,1,n)**。
- `MuonAdamW(torch.optim.Optimizer)`（L212-349，单 GPU，launch 用）：
  - `__init__`：建 10 个 0-D CPU 张量（AdamW 6 + Muon 4）供 fused 内核。
  - `_step_adamw(group)`（L248-282）：逐参数；`p.grad is None` 跳过；状态**惰性初始化**
    （exp_avg/exp_avg_sq zeros_like + step 计数）；填 0-D 张量 → `adamw_step_fused`。
  - `_step_muon(group)`（L284-339）：组级状态存第一个参数；`stacked_grads/stacked_params = torch.stack(...)`；
    **内部 lr 修正** `lr × max(1, m/n)^0.5`（tall 矩阵加速）；`red_dim = -1 if m≥n else -2`；
    fused 后 `torch._foreach_copy_` 拷回原参数。
  - `step()`（L341-349）：按 `group['kind']` 分派 adamw/muon。
- `DistMuonAdamW`（L355-596）：ZeRO-2 风格 3 阶段异步（①启动全部 reduce_scatter/all_reduce →
  ②等待+计算+启动 gather → ③等待 gather 拷回），梯度 all_reduce 取平均、优化器状态分片；
  **launch 单进程不激活**——一句话标注即可。

`scaler`（脚本 L558）：`torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None` →
launch 下 **None**（fp32 不需要梯度缩放）。

##### 2.3.3-09 数据加载器 —— `_document_batches / refill_buffer / tokenizing_distributed_data_loader_with_state_bos_bestfit / ..._bos_bestfit`（nanochat/dataloader.py）—— 完整展开（首次使用）

核心思想（模块 docstring）：每行以 `<|bos|>` 开头（BOS 对齐）；**Best-Fit 打包**：先找"能完整放下的
最大文档"，找不到就裁剪最短文档填满剩余 → **100% 利用率（无 padding），代价是 ~35% token 被裁剪**。

- `_document_batches(split, resume_state_dict, tokenizer_batch_size)`（L30-77）——无限文档流：

```
ddp, rank, _, world = get_dist_info()            ↩ §2.3.3-01（launch: rank0 单进程）
parquet_paths = list_parquet_files(warn_on_legacy=rank0 且 train)   ↩ §2.1.3-②
train = 除最后分片；val = 仅最后分片（shard_06542）
续训状态 resume_state_dict = {pq_idx, rg_idx, epoch}（None → 从头）
while True:                                       # 无限迭代（多 epoch）
    for 每个 parquet 文件:
        ParquetFile → row_group 按 DDP rank 交错（rank0 顺序读全部 rg）
        batch = rg.column('text').to_pylist()
        每 tokenizer_batch_size=128 篇 yield (batch, (pq_idx, rg_idx, epoch))
```

- `tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, B, T, split, tokenizer_threads=4,
  tokenizer_batch_size=128, device, resume_state_dict=None, buffer_size=1000)`（L80-174）：

```
row_capacity = T + 1 = 513            # 多 1 列给 y = x 右移 1
batches = _document_batches(...)      # 文档批流
bos_token = tokenizer.get_bos_token_id()   ↩ §2.5.3-④（launch: 0）
doc_buffer = []                       # 已分词文档池（≥1000 条）

def refill_buffer():                  # 闭包（L115-121）
    doc_batch, (pq_idx, rg_idx, epoch) = next(batches)       # 128 篇原始文本
    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)   ↩ §2.1.3-⑥ 批量路径
    for tokens in token_lists:  doc_buffer.append(tokens)    # 每条 = [0(bos), ...doc tokens]
```

  预分配缓冲区（launch B=1, T=512, device=cpu）：

| 缓冲区 | 形状 | dtype | 作用 |
|---|---|---|---|
| `row_buffer` | `(B, T+1)` = `(1, 513)` | int64 | 逐行构建（x/y 共享，y 是 x 右移 1） |
| `cpu_buffer` | `(2·B·T)` = `(1024,)` | int64 | CPU 暂存（use_cuda=False → 不 pin） |
| `gpu_buffer` | `(2·B·T)` = `(1024,)` | int64 | 设备侧持久缓冲区（launch 下 device=cpu，是与 cpu_buffer 分开的另一块内存） |
| `inputs / targets` | `(B, T)` = `(1, 512)` | int64 | gpu_buffer 的视图切片：x=前 512，y=后 512 |

  Best-Fit 打包主循环（每行）：

```
for row_idx in range(B):
    pos = 0
    while pos < row_capacity(513):
        while len(doc_buffer) < 1000:  refill_buffer()          # 保持池子足够大
        remaining = 513 - pos
        best = 池中"能完整放入"的最长文档                        # 优先无裁剪
        if 找到: 写入 row_buffer[row, pos:pos+len]; pos += len
        else:    裁剪池中最短文档填满 remaining                  # 精确填满，无 padding
```
```
  文档 A(200)  文档 B(80)  [裁剪] 文档 C(233→232)
  [bos|A.............|bos|B......|bos|C.......(截断)]   ← 每段都以 <|bos|> 开头
  0                200       280              512/513
```
```
cpu_inputs.copy_(row_buffer[:, :-1]);  cpu_targets.copy_(row_buffer[:, 1:])    # x=前512, y=后512
state_dict = {"pq_idx","rg_idx","epoch"}                                        # 断点续训状态
gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)                             # 单次 HtoD
yield inputs, targets, state_dict
```

- `tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs)`（L176-179）：薄封装，yield 时**去掉
  state_dict** 只给 `(inputs, targets)` —— eval 用（§2.3.3-12 / §2.4.3-⑤）。

脚本侧（L565-568）：`train_loader = ..._with_state...(tokenizer, 1, 512, "train", device)`；
`build_val_loader = lambda: ..._bos_bestfit(tokenizer, 1, 512, "val", device)`（eval 钩子现建）；
`x, y, dataloader_state_dict = next(train_loader)` 预取第一批 `(1,512)`。

##### 2.3.3-10 训练步数与三个调度器（脚本 L576-638，本文件）—— 完整展开

```
num_iterations: --num-iterations 100 优先（> target_flops > target_param_data_ratio 互斥优先级）
total_tokens = 32768 × 100 = 3,276,800 token
Token : Scaling params = 3,276,800 / 2,490,416 ≈ 1.32   （Chinchilla ≈ 20，冒烟跑故意小）
```

- `get_lr_multiplier(it)`（L598-611）：

```
warmup_iters  = 10
warmdown_iters = round(0.65 × 100) = 65   → warmdown 起点 = 100−65 = 35
it < 10:            (it+1)/10                 线性预热
it ≤ 35:            1.0                       平台期
否则:               (100−it)/65·1 + (1−…)·0.05  线性降到 final_lr_frac=0.05
```

- `get_muon_momentum(it)`（L615-629）：`it<400` 时 `0.85→0.97` 线性（100 步全程在此段）；`≥warmdown_start(35)`
  时 `0.97→0.90`；否则 0.97。→ 100 步运行内：前 35 步 0.85→0.97，后 65 步 0.97→0.90。
- `get_weight_decay(it)`（L633-638）：`weight_decay_scaled × 0.5 × (1 + cos(π·it/100))` → 余弦衰减到 0。

```
lrm
1.0 ┤      ┌──────────────────┐
0.5 ┤     ╱                    ╲
0.0 ┴────╱                      ╲───►  0.05
    0   10                     35   100  step
```

##### 2.3.3-11 训练循环骨架与梯度累积（脚本 L670-813，本文件）

```
grad_accum_steps = 32768 // (1×512×1) = 64          # total_batch_size / (device_batch × seq_len × world)

while True:
    last_step = (step == 100)
    [钩子区·每步开头先检查]
      ① eval 钩子:  --eval-every 50 → step 0/50/100 → evaluate_bpb（§2.3.3-12）
      ② core 钩子: --core-metric-every -1 → 关闭（死分支 → §2.4.3-⑥）
      ③ sample 钩子: --sample-every 50 → step 0/50/100（仅主进程）→ Engine（§2.3.3-13）
      ④ save 钩子: --save-every -1 → 仅 last_step → save_checkpoint（§2.3.3-14）
    if last_step: break
    [微步区]
    for micro_step in range(64):
        loss = model(x, y)                    # 前向 → §2.3.3-15 七步大图
        (loss / 64).backward()                # 每 .backward() 是梯度累加 → 除以 64 归一化
        x, y, dataloader_state_dict = next(train_loader)   # 前向/反向期间预取下一批
    [优化器区]
    lrm/muon_momentum/muon_wd = 三个调度器(step)          # §2.3.3-10
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] × lrm
        muon 组: group["momentum"] = muon_momentum; group["weight_decay"] = muon_wd
    optimizer.step()                          # MuonAdamW.step ↩ §2.3.3-08
    model.zero_grad(set_to_none=True)
    [日志区] EMA/ETA/MFU/打印/wandb/GC           # §2.3.3-16
    step += 1
```

> **autograd 梯度追踪 = "算梯度"；优化器 = "调参数"，两步配合才是训练**。
> - ① 前向建图：`nn.Parameter` 默认 `requires_grad=True` → 经过它的每个算子被记录进计算图（grad_fn 链），
>   且必须**保存中间激活**供反向使用（logits (1,512,16384) ≈33.5MB 等是显存大头）；
> - ② `loss.backward()`：链式法则沿图回传，把 `∂L/∂θ` 填进每个 `param.grad`；
> - ③ `optimizer.step()`：**优化器**读 `.grad` 并结合自身状态（AdamW 动量/二阶矩、Muon 正交化，§2.3.3-08）
>   算出更新量应用——调参数的活是优化器干的，不是 autograd；
> - ④ `zero_grad(set_to_none=True)`：清空 `.grad`，等下个微步。
> 所以"不需要梯度"的场景（初始化/评估/推理）用 `@torch.no_grad()`/`@torch.inference_mode()` 关掉追踪：
> 不建图、不存中间量 → 省内存加速（§2.3.3-06 的 init_weights 正是如此）。

##### 2.3.3-12 验证钩子 —— `evaluate_bpb`（nanochat/loss_eval.py L9-69）—— 完整展开（首次使用）

**BPB（bits per byte）** = 与词表大小无关的损失度量（普通 mean loss 换词表就不可比）：

```
BPB = Σ_tokens loss(t)·[bytes(t)>0] / (ln2 · Σ_tokens bytes(t))
```

```
@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats  = tensor(0.0, fp32, model.get_device())    # 累计 nats
    total_bytes = tensor(0,   int64, model.get_device())
    for _ in range(steps):
        x, y = next(batches)                               # launch: (1,512)
        loss2d = model(x, y, loss_reduction='none')        # 逐 token 损失 (1,512)
        loss2d, y = loss2d.view(-1), y.view(-1)            # 展平 (512,)
        if (y < 0).any():                                  # 慢路径: 有 ignore_index=-1
            valid  = y >= 0;  y_safe = where(valid, y, 0)  # 负索引不能查 token_bytes
            num_bytes2d = where(valid, token_bytes[y_safe], 0)
        else:                                              # 快路径（launch 预训练走这里）
            num_bytes2d = token_bytes[y]                   # (512,) 每目标 token 的字节数
        total_nats  += (loss2d × (num_bytes2d > 0)).sum()  # 特殊 token 字节=0 → 不计
        total_bytes += num_bytes2d.sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1   # launch=1
    if world_size > 1:  all_reduce 两个累加器
    bpb = total_nats.item() / (math.log(2) × total_bytes.item())   # bytes=0 → inf
```

launch 数值：`eval_steps = 65536 // (1×512×1) = 128` 步/次；val_loader 由 `build_val_loader()`
（`..._bos_bestfit`，↩ §2.3.3-09）现建；在 `disable_fp8(model)` 上下文里跑（launch 下空转，§2.3.3-06）。
`token_bytes` 正是 1b 生成的 `(16384,) int32`（§2.1.3-⑦）。
实测 meta_000100.json：**val_bpb = 1.9506**。

##### 2.3.3-13 采样钩子 —— `Engine`（nanochat/engine.py）—— 完整展开（首次使用）

**模块职责**：高效推理引擎（KV Cache + 批量采样 + 工具调用状态机），复杂度 O(N²)→O(N)。

- `Engine.__init__(model, tokenizer)`（L235-239）：只存两引用（tokenizer 供工具调用编解码）。
- `KVCache`（L102-180，FA3 原生布局 **`(B,T,H,D)`，无需 transpose**）：

| 成员 | 形状 | 说明 |
|---|---|---|
| `k_cache / v_cache` | `(n_layers, B, T_max, KV, D)` | 预分配缓存 |
| `cache_seqlens` | `(B,) int32` | 每行当前长度（FA3 要求 int32） |
| `prev_embedding` | `(B,1,C)` | 上一 token 归一化嵌入（供 Smear，decode 时用） |

  方法：`reset()`（清零+清 prev）；`get_pos()`（取 `cache_seqlens[0]`）；`get_layer_cache(i)`
  （返回第 i 层 (k,v) 视图）；`advance(n)`（缓存位置 +n，由最后一层注意力处理后调用）；
  `prefill(other)`（把另一个缓存的 KV 拷进来 + `prev_embedding.expand(B,-1,-1).clone()` 广播——
  batch=1 prefill → batch=N decode 的桥）。
- `sample_next_token(logits (B,V), rng, temperature, top_k)`（L184-204）→ `(B,1)`：
  温度 0 → argmax；top_k>0 → topk(k) → ÷温度 → softmax → multinomial → `idx.gather`；否则全词表 softmax 采样。
- `RowState`（L208-224）：每行状态 `current_tokens / forced_tokens(deque) / in_python_block /
  python_expr_tokens / completed`——工具调用强制注入 token 的队列机制。
- 计算器三函数（L35-99）：`timeout`（SIGALRM 上下文管理器；**Windows 无 SIGALRM** → 抛异常被
  `eval_with_timeout` 的 try/except 吞掉 → 计算器不可用，返回 None——不影响生成）；`eval_with_timeout`
  （`eval(formula, {"__builtins__": {}}, {})` 空 builtins 防注入）；`use_calculator(expr)`（纯数学禁 `**`；
  字符串白名单字符 + 危险模式黑名单 + 仅允许 `.count()`）。
- `Engine.generate(tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42)`（L241-352）：

```
① Prefill（batch=1，一次前向填满 KV）
   kv_cache_prefill = KVCache(B=1, seq_len=len(tokens), KV头=4, D=32, 2层, fp32)
       → k_cache/v_cache (2, 1, T_conv, 4, 32)
   ids = tensor([[tokens]])                          (1, T_conv)
   logits = model.forward(ids, kv_cache=prefill)     (1, T_conv, 16384)
   logits = logits[:, -1, :].expand(num_samples, -1) → (N, 16384)   只取末位置并复制
② 广播（多样本并行）
   kv_cache_decode = KVCache(B=N, seq_len=T_conv+max_tokens, ...)
       → k_cache/v_cache (2, N, T_conv+256, 4, 32)
   kv_cache_decode.prefill(kv_cache_prefill)         # 拷贝 KV 与 prev_embedding（expand 广播）
   del kv_cache_prefill
③ 初始化 row_states = [RowState(tokens.copy()) × N]
④ Decode 循环（每步）
   next_ids = sample_next_token(logits, rng, temperature, top_k)   (N,1)
   逐行: 有 forced_tokens → 强制注入（mask=0），否则用采样（mask=1）
         current_tokens.append; assistant_end/bos → completed=True
         工具状态机: 遇 python_start → 开块收集表达式;
                    遇 python_end → decode 表达式 → use_calculator → 结果 encode →
                    forced_tokens += [output_start, ...结果, output_end]
   yield (token_column, token_masks)                 # 流式：每步一列
   ids = tensor(token_column).unsqueeze(1)           (N,1)
   logits = model.forward(ids, kv_cache=decode)[:, -1, :]   (N, 16384)
   终止: max_tokens 或 全部行 completed
```

- `generate_batch(tokens, num_samples, **kwargs)`（L354-381）：非流式收集版——把 `generate` 的流
  收成 `results/masks`（终止 token 不包含）。base_train/base_eval 的采样都走它。

base_train 采样钩子（L719-736）：7 句英文提示 → `tokenizer(prompt, prepend="<|bos|>")`
（`__call__`=encode ↩ §2.1.3-⑥）→ `engine.generate_batch(tokens, num_samples=1, max_tokens=16,
temperature=0)`（贪婪）→ `tokenizer.decode(sample[0])` 打印。用 `orig_model`（避免重编译）。
**完整对话版 prefill/decode 大图见 §2.6.3-④。**

##### 2.3.3-14 保存钩子 —— `save_checkpoint`（nanochat/checkpoint_manager.py L49-78）—— 完整展开（首次使用）

```
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        torch.save(model_data,  f"{dir}/model_{step:06d}.pt")    # 模型参数（仅 rank0）
        json.dump(meta_data,    f"{dir}/meta_{step:06d}.json", indent=2)   # 元数据
    if optimizer_data is not None:
        torch.save(optimizer_data, f"{dir}/optim_{step:06d}_rank{rank}.pt") # 优化器分片（每 rank）
```

base_train 传入：`orig_model.state_dict()`、`optimizer.state_dict()`、meta 字典（`step / val_bpb /
model_config / user_config / device_batch_size / max_seq_len / total_batch_size / dataloader_state_dict /
loop_state{min_val_bpb, smooth_train_loss, total_training_time}`）、`rank=ddp_rank`。
实测产物：`model_000100.pt` 26.7 MB（≈6.68M 参数×fp32 4B）、`optim_000100_rank0.pt` 51.9 MB
（AdamW 双动量状态为主）、`meta_000100.json`。

##### 2.3.3-15 前向核心 —— `GPT.forward` 七步 + `CausalSelfAttention/MLP/Block/apply_rotary_emb` + `flash_attention`（nanochat/gpt.py + nanochat/flash_attention.py）—— 完整展开（首次使用）

**d2 全形状数据流图（B=1, T=512）**：

```
 idx (1,512) int64
   │ ① RoPE 缓存校验: T=512 ≤ 5120; 截取 cos/sin (1,5120,1,16) 的 [:,0:512] → (1,512,1,16)
   ▼ ② 嵌入
 x = wte(idx)                       (1,512) → (1,512,128)
 x = x.to(float32); x = norm(x)     (1,512,128)    [x0 = 保存副本]
   │ ③ Smear (训练路径, T>1)
   │    gate = smear_lambda·σ( smear_gate( x[:,1:,:24] ) )    smear_gate: (1,24) → (1,511,1)
   │    x = cat([x[:,:1], x[:,1:] + gate·x[:,:-1]], dim=1)    x_t += gate_t · x_{t-1}  (1,512,128)
   ▼ ④ 逐层 for i in 0,1:
   │    x = resid_lambdas[i]·x + x0_lambdas[i]·x0    (1,512,128)   标量广播
   │    ve = value_embeds["1"](idx).to(fp32)         仅 i==1 → (1,512,128)   [i==0: None]
   │    ┌─ Block ─────────────────────────────────────────────────────┐
   │    │ Attn:  n = norm(x)                                  (1,512,128)
   │    │        q = c_q(n) → (1,512,128) → view (1,512,4,32)
   │    │        k = c_k(n) → view (1,512,4,32)
   │    │        v = c_v(n) → view (1,512,4,32)
   │    │        [i==1] gate = 3·σ(ve_gate(x[...,:12]))   ve_gate (4,12) → (1,512,4)
   │    │               v = v + gate.unsqueeze(-1)·ve     (1,512,4,32)
   │    │        RoPE: x1,x2 = x[...,:16], x[...,16:]     各 (1,512,4,16)
   │    │              y1 = x1·cos + x2·sin ; y2 = -x1·sin + x2·cos → cat → (1,512,4,32)
   │    │        q,k = norm(q)·1.2 , norm(k)·1.2           QK-Norm（scale 拆成 q、k 各 1.2）
   │    │        y = flash_attn_func(q,k,v, causal=True, window_size=window_sizes[i])
   │    │            SDPA 回退: (1,512,4,32) →transpose→ (1,4,512,32) → y (1,512,4,32)
   │    │        y.view(1,512,128) → c_proj → attn_out    (1,512,128)
   │    │ MLP:   n = norm(x) → c_fc (512,128) → (1,512,512) → ReLU² → (1,512,512)
   │    │        → c_proj (128,512) → (1,512,128)
   │    │ 残差:  x = x + attn_out ; x = x + mlp_out        (1,512,128)   [Pre-LN 两段]
   │    └─────────────────────────────────────────────────────────────┘
   │    i == n_layer//2 == 1 时: x_backout = x            (1,512,128)   [d2: 最后一层]
   ▼ ⑤ Backout
 x = x - backout_lambda·x_backout                          (1,512,128)   去除低层拼写/词法特征
   ▼ ⑥ lm_head
 x = norm(x)                                               (1,512,128)
 logits = lm_head(x)  (16384,128) → (1,512,16384)
 logits = logits[..., :16384].float()                      (1,512,16384) fp32
 logits = 15·tanh(logits/15)                               softcap
   ▼ ⑦ loss
 loss = CE(logits.view(512, 16384), y.view(512), ignore_index=-1, reduction='mean')   → 标量 (fp32)
```

配套方法：

- `apply_rotary_emb(x (B,T,H,D), cos, sin)`（gpt.py L89-97）：最后维对半切 `x1,x2`，2D 旋转
  `y1 = x1·cos + x2·sin; y2 = −x1·sin + x2·cos`，`cat` 回来——Q·K 内积自动含相对位置。
- `CausalSelfAttention.forward(x, ve, cos_sin, window_size, kv_cache)`（L132-188）：见上图；
  推理分支（`kv_cache` 非 None）走 `flash_attn_with_kvcache`，最后一层处理后 `kv_cache.advance(T)`。
- `MLP.forward`（L202-208）：`c_fc → F.relu(x).square() → c_proj`（ReLU² 是廉价的 GeLU 门控近似）。
- `Block.forward`（L222-227）：`x = x + attn(norm(x), ...); x = x + mlp(norm(x))`。
- `GPT.forward` 推理分支要点（chat_cli 用）：`T0 = kv_cache.get_pos()` 偏移 RoPE；Smear 分三态
  （prefill T>1 同训练 / decode T==1 用 `kv_cache.prev_embedding` 且存新 `x[:, -1:]`）；
  `targets is None` → 直接返回 logits。
- `GPT.generate(tokens, max_tokens, temperature, top_k, seed)`（L654-689）：朴素 O(N²) 无 KV Cache 的
  流式生成（batch=1、Python list）——**launch 链路不用**（仅 engine.py `__main__` 自测对照），标注即可。

**`nanochat/flash_attention.py` —— 完整展开**（统一注意力接口，FA3/SDPA 自动切换）：

- `_load_flash_attention_3()`（L22-41）：无 CUDA → None；CUDA 且 SM 大版本 == 9（Hopper）→
  `kernels.get_kernel('varunneal/flash-attention-3').flash_attn_interface`；否则 None。
  **launch CPU → None → `HAS_FA3=False`**。
- `_resolve_use_fa3()`（L51-67）：手动覆盖 > `HAS_FA3 且 COMPUTE_DTYPE==bf16` > False。
  **launch → `USE_FA3=False`（SDPA 回退）**。
- `_sdpa_attention(q,k,v (B,H,T,D), window_size, enable_gqa)`（L75-114）三路径：
  ① 完整上下文且 Tq==Tk → `F.scaled_dot_product_attention(..., is_causal=True)`；
  ② 单 token 生成（Tq==1）→ 裁剪窗口内 KV，`is_causal=False`；
  ③ 分块推理/滑窗训练 → 显式 bool 掩码 `(col ≤ row) & ((row−col) ≤ window)`。
- `flash_attn_func(q,k,v,causal,window_size)`（L119-140）：FA3 直传；SDPA 需
  `transpose(1,2)` 转 `(B,H,T,D)`，算完转回。
- `flash_attn_with_kvcache(q,k_cache,v_cache,k,v,cache_seqlens,...)`（L143-189）：FA3 原地更新缓存；
  SDPA 手动写 `k_cache[:, pos:pos+T] = k` → 取 `[:end_pos]` 全量算。
- **注意**：launch 2a 用默认 `window_pattern="SSSL"` → SDPA 回退用显式 mask 实现滑窗（功能正确，
  只是无硬件加速），所以脚本会打"建议 --window-pattern L"的警告（§2.3.3-02）。

##### 2.3.3-16 日志 / GC / 收尾（脚本 L811-908，本文件）

- 日志（每步）：`smooth_train_loss = 0.9·旧 + 0.1·train_loss`（EMA），打印时去偏
  `/(1−0.9^(step+1))`；`tok_per_sec = 32768/dt`；`flops_per_sec`；`mfu = 100·flops/(inf·1) → 0%`
  （CPU 无峰值表）；10 步后按平均步时算 ETA；`epoch/pq/rg` 来自 dataloader_state_dict；
  `step%100` 时 `wandb_run.log(...)`（DummyWandb no-op ↩ §2.3.3-01）。
- GC 干预（L864-869）：第 0 步 `gc.collect()+gc.freeze()+gc.disable()`（初始化垃圾冻结出扫描区），
  之后每 5000 步 `gc.collect()`——避免 GC 每 ~500ms 扫循环引用。
- 收尾：打印峰值内存/总时间/min_val_bpb；`get_report().log(section="Base model training / 基座模型训练",
  data=[user_config, 训练设置统计, 训练结果统计])` ↩ 完整展开见 §2.1.3-⑧ →
  `data/report/base-model-training---基座模型训练.md`（slugify 把 `" / "` 变成 `"---"`）。
- `compute_cleanup()`（common.py L271-274）—— 完整展开：`if is_ddp_initialized():
  dist.destroy_process_group()`。launch 单进程 → no-op。`wandb_run.finish()`（Dummy no-op）。

#### 2.3.4 数据流与梯度累积（launch 2a 数值）

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
  shard_00000..00007.parquet（train）→ row_group → 128 篇/批 → encode(prepend=<|bos|>, 4线程)
  → doc_buffer(≥1000) → Best-Fit 打包进 row_buffer (1,513)
  → x = row[:, :-1] (1,512), y = row[:, 1:] (1,512)
  → cpu_buffer (1024,) → 单次拷贝 → yield
```

> **B/T 与"内部张量"的关系**：**权重形状**（(16384,128)、(128,128) 等）只由
> vocab/n_embd/n_head/n_kv_head/head_dim/n_layer 决定，**与 B、T 无关**——换 batch 或序列长度，
> 权重矩阵不变。但**激活张量**完全由 B、T 决定：`x (B,T,128)`、MLP 中间 `(B,T,512)`、
> logits `(B,T,16384)`（最大激活，d2: 1×512×16384×4B ≈ 33.5MB）、注意力 O(B·T²)、KV Cache `(2,B,T,4,32)`。
> 所以"显卡强 → 加大 B/T"要分两种：
> - **B（device-batch-size）是纯运行时参数**：加大 = 每微步处理更多 token、微步数变少
>   （`total_batch_size` 固定时**梯度累积自动补偿，训练效果不变**）；显存不够就调小 B——本节的 64 微步就是这么来的。
> - **T（max-seq-len）会改模型配置**：T 是 `GPTConfig.sequence_len` 字段——改它影响 window_sizes 推导
>   （§2.3.3-05）、RoPE 缓存长度、检查点 meta 记录；且注意力 **O(T²)**，T 翻倍注意力开销约 4 倍。
> - 多卡"分片"：`world_tokens_per_fwdbwd = device_batch × seq_len × world_size`，world_size 越大
>   每 rank 微步越少（launch 单进程 world_size=1）。

#### 2.3.5 循环内钩子总表与实测产物

| 钩子 | 参数 | 触发 | 动作 |
|---|---|---|---|
| 验证 BPB | `--eval-every 50 --eval-tokens 65536` | step 0/50/100 | `evaluate_bpb`（§2.3.3-12），val loader B=1，128 步 |
| CORE | `--core-metric-every -1` | 关闭 | —（完整展开见 §2.4.3-⑥） |
| 采样 | `--sample-every 50` | step 0/50/100（仅主进程） | 7 句提示 × Engine 贪婪生成 16 token（§2.3.3-13） |
| 保存 | `--save-every -1` | 仅结束 | `save_checkpoint` → model/optim/meta_000100（§2.3.3-14） |

实测 `data/base_checkpoints/d2/meta_000100.json`：`val_bpb=1.9506`、`smooth_train_loss=6.1369`、
`total_training_time≈250.4s`、`dataloader_state_dict={pq_idx:0, rg_idx:9, epoch:1}`
（即 100 步只吃到第 1 个 shard 的第 9 个 row_group）。上述为**改动前（1 头）**运行的实测；
切到方案 C（4 头）后参数量/FLOPs 仅微变（§2.3.3-07），指标量级相当，重跑后 meta 自动更新。

### 2.4 `3a. Base Eval` → `scripts/base_eval.py`

#### 2.4.1 定位与 CLI

```jsonc
"args": [
    "--model-tag", "d2",            // 与 2a 的 --run 对应 → base_checkpoints/d2/
    "--step", "100",                // 评估第 100 步检查点
    "--device-type", "cpu",
    "--device-batch-size", "4",     // BPB 评估批大小
    "--eval", "core,bpb,sample",    // 三种评估全开
    "--max-per-task", "20",         // 每个 CORE 任务只取 20 题
    "--split-tokens", "8192"        // BPB 每个 split 只用 8192 token → 4 步（§2.4.3-⑤）
]
```

本脚本**有 `main()`**（`if __name__ == "__main__": main()`）。也定义了被 2a 跨脚本 import 的
`evaluate_core`（base_train 第 47 行 `from scripts.base_eval import evaluate_core`——2a 里因
`--core-metric-every -1` 是死分支，真正执行在本节）。

#### 2.4.2 调用树总图

```
scripts/base_eval.py main()
│
├─ [1] argparse → eval_modes = {"core","bpb","sample"} 校验
├─ [2] autodetect_device_type / compute_init          ↩ §2.3.3-01
├─ [3] is_hf_model = (--hf-path != None) → False（launch）→ 死分支见 ③-2
├─ [4] load_model("base", device, phase="eval", model_tag="d2", step=100)
│      ← nanochat/checkpoint_manager.py，③-3 完整展开（首次使用）
│        ├─ load_model_from_dir → find_largest_model / find_last_step → build_model（六步）
│        ├─ load_checkpoint → torch.load(model_000100.pt / meta_000100.json)
│        ├─ _patch_missing_config_keys / _patch_missing_keys（向后兼容）
│        ├─ GPT / GPTConfig（gpt.py，↩ §2.3.3-04/05）→ meta 建壳 → to_empty → init_weights
│        └─ get_tokenizer        ↩ §2.2.3-③
├─ [5] token_bytes = get_token_bytes(device)          ↩ §2.3.3-03
├─ [6] sample 模式: Engine(model, tokenizer) + generate_batch   ↩ §2.3.3-13
├─ [7] bpb 模式: tokenizing_distributed_data_loader_bos_bestfit ↩ §2.3.3-09
│      └─ evaluate_bpb(model, loader, steps, token_bytes)       ↩ §2.3.3-12
├─ [8] core 模式: evaluate_core(model, tokenizer, device, max_per_task=20)
│      ← 本文件 + nanochat/core_eval.py，③-6 完整展开（首次使用）
│        ├─ download_file_with_lock(EVAL_BUNDLE_URL, ..., postprocess_fn=place_eval_bundle)
│        │    ← nanochat/common.py，③-6 完整展开
│        │      ├─ FileLock（第三方 filelock） + urllib.request
│        │      └─ place_eval_bundle: zipfile / tempfile / shutil
│        ├─ yaml.safe_load(core.yaml) → icl_tasks（实测 22 个）
│        ├─ evaluate_task → evaluate_example → render_prompts_{mc,schema,lm}（jinja2）
│        │    → batch_sequences_{mc,schema,lm} → stack_sequences → forward_model
│        └─ csv.DictReader(eval_meta_data.csv) → random baselines → centered
├─ [9] CSV 写出 → data/base_eval/base_model_000100.csv
└─ [10] get_report().log          ↩ §2.1.3-⑧ ； compute_cleanup  ↩ §2.3.3-16
```

#### 2.4.3 逐方法展开

##### ③-1 `main()` 骨架（L223-382，本文件）

argparse（8 个参数，默认值见 2.4.1）→ `eval_modes = set(m.strip() for m in args.eval.split(','))`
并校验 ⊆ {core,bpb,sample} → `autodetect_device_type/compute_init`（↩ §2.3.3-01，返回
`(False,0,0,1,cpu)`）→ 加载模型（③-3）→ 依序跑 sample → bpb → core → CSV → report → cleanup。

##### ③-2 HuggingFace 死分支 —— `ModelWrapper / load_hf_model / get_hf_token_bytes`（L66-121，本文件；launch `--hf-path` 未设，不执行）

- `ModelWrapper(model, max_seq_len)`：给 HF 模型套 nanochat 兼容接口。`__call__(input_ids, targets,
  loss_reduction)`：`logits = model(input_ids).logits` → 无 targets 返回 logits；有则
  `CE(logits.view(-1, V), targets.view(-1), ignore_index=-1, reduction=...)`（与 GPT.forward ⑦同款）。
  `get_device()`：`next(model.parameters()).device`。`max_seq_len` 供 core_eval 截断长序列（③-6）。
- `load_hf_model(hf_path, device)`：`transformers.AutoModelForCausalLM.from_pretrained` [第三方] →
  `.to(device).eval()`；`"gpt2"` 在路径里 → `max_seq_len=1024`；
  `HuggingFaceTokenizer.from_pretrained(hf_path)`（tokenizer.py L59-64：`tokenizers.Tokenizer.from_pretrained`
  [第三方 HF tokenizers]——与 RustBPETokenizer 并存但 launch 不用的那套实现）。
- `get_hf_token_bytes(tokenizer, device)`：`torch.zeros(vocab, int64)` 循环 `decode([id])` 填 UTF-8 字节数
  → `(vocab,) int64`（BPB 分母的 HF 版）。

##### ③-3 模型加载 —— `load_model / load_model_from_dir / find_largest_model / find_last_step / build_model / load_checkpoint / _patch_missing_*`（nanochat/checkpoint_manager.py）—— 完整展开（首次使用）

- `load_model(source, *args, **kwargs)`（L218-233）：

```
model_dir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[source]
checkpoints_dir = get_base_dir()/model_dir          ↩ §2.1.3-④
return load_model_from_dir(checkpoints_dir, *args, **kwargs)
```

- `load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None)`（L190-216）：
  `model_tag=None` → `find_largest_model`；`step=None` → `find_last_step`；→ `build_model`。
  - `find_largest_model`（L157-173）：子目录名匹配 `d(\d+)` 取 depth 最大；无匹配 → mtime 最新。
  - `find_last_step`（L176-185）：`glob(model_*.pt)` 提取最大步数。
- `load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0)`（L80-106）：
  `torch.load(model_{step:06d}.pt, map_location=device)`；可选 optim 分片；`json.load(meta_{step:06d}.json)`。
- `build_model(checkpoint_dir, step, device, phase)`（L109-154）**六步**：

```
① model_data, _, meta = load_checkpoint(...)
② CPU/MPS: bf16 张量 → .float()               # CPU 推理不支持 bf16
③ 去前缀 {k.removeprefix("_orig_mod."): v}     # torch.compile 保存的 state_dict 带前缀
④ model_config = GPTConfig(**meta["model_config"])   # d2(方案C): n_layer=2,n_embd=128,n_head=4,vocab=16384...
   _patch_missing_config_keys: 旧检查点无 window_pattern → 补 "L"
   _patch_missing_keys:        旧权重缺 resid_lambdas → ones(n_layer); 缺 x0_lambdas → zeros(n_layer)
⑤ with torch.device("meta"): model = GPT(model_config)   # 建壳（↩ §2.3.3-04/05）
   model.to_empty(device); model.init_weights()           # 分配 + 临时初始化
   model.load_state_dict(model_data, strict=True, assign=True)   # 覆盖为保存的权重
⑥ phase=="eval" → model.eval() 否则 model.train()
   tokenizer = get_tokenizer()  ↩ §2.2.3-③
   assert tokenizer.get_vocab_size() == model_config.vocab_size   # 词表一致性（16384 ✓）
return (model, tokenizer, meta)
```

- `log0`（L23-26）：rank0 专用 logger。`load_optimizer_state`（L235-255）：只加载某 rank 的 optim 分片，
  **launch 链路不调用**（chat_rl 用），一句。

launch 结果：`model`（d2，eval 模式）、`tokenizer`（vocab 16384）、`meta`（`sequence_len=512`、`step=100`）。

##### ③-4 sample 采样模式（L279-312，本文件）

`Engine(model, tokenizer)`（↩ §2.3.3-13）：
- 7 句条件采样：`tokenizer(prompt, prepend="<|bos|>")` → `engine.generate_batch(tokens, num_samples=1,
  max_tokens=16, temperature=0)`（贪婪）→ `tokenizer.decode(sample[0])` 打印。
- 8 条无条件采样：`tokenizer("", prepend="<|bos|>")` → `generate_batch(..., num_samples=8,
  max_tokens=128, temperature=1.0)` → 逐条 decode。
- HF 模型分支：打印"跳过（不支持）"。

##### ③-5 BPB 评估模式（L316-332，本文件）

```
tokens_per_step = device_batch_size(4) × sequence_len(512) × world(1) = 2048
split_tokens 8192 % 2048 == 0 → steps = 8192/2048 = 4 步/split
for split in ["train", "val"]:
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 4, 512, split, device)  ↩ §2.3.3-09
    bpb = evaluate_bpb(model, loader, 4, token_bytes)                                          ↩ §2.3.3-12
```

x,y `(4,512)` → loss2d `(4,512)` → 按 token_bytes 加权 → BPB。结果进 `bpb_results{train,val}` → report。

##### ③-6 CORE 评估 —— `evaluate_core / place_eval_bundle / download_file_with_lock` + `core_eval.py` 全体 —— 完整展开（首次使用）

**`evaluate_core(model, tokenizer, device, max_per_task=-1)`（base_eval L144-217，本文件）**：

```
eval_bundle_dir = base_dir/eval_bundle
if 不存在: download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
config = yaml.safe_load(core.yaml);  tasks = config['icl_tasks']        # 实测 22 个任务
random_baselines = csv.DictReader(eval_meta_data.csv) → {任务: 随机基线准确率}
for task in tasks:
    label; task_meta = {task_type, dataset_uri, num_fewshot(=task['num_fewshot'][0]),
                        continuation_delimiter(默认 ' ')}
    data = [json.loads(line) for line in open(eval_data/dataset_uri)]   # 任务 jsonl
    shuffle_rng = random.Random(1337); shuffle_rng.shuffle(data)        # 一致子采样
    max_per_task=20 → data = data[:20]
    accuracy = evaluate_task(model, tokenizer, data, device, task_meta) # core_eval.py ↓
    centered = (accuracy − 0.01·baseline) / (1 − 0.01·baseline)         # 中心化（对随机基线归一）
core_metric = mean(centered) → {"results", "centered_results", "core_metric"}
```

- `download_file_with_lock(url, filename, postprocess_fn)`（common.py L96-141）—— 完整展开：目标已存在 →
  直接返回；`FileLock(path+".lock")`（[第三方 filelock] 多 rank 互斥）→ 锁内二次检查 → `urllib.request.urlopen
  (timeout=60)` 下载，失败最多 5 次、指数退避 `2^attempt` 秒 → 写文件 → 调 `postprocess_fn`。
- `place_eval_bundle(file_path)`（L130-141）：`TemporaryDirectory` + `zipfile.ZipFile.extractall` +
  `shutil.move` → `data/eval_bundle/`（core.yaml + eval_data/ + eval_meta_data.csv）。

**`nanochat/core_eval.py` 全体 —— 完整展开**（DCLM 论文 CORE 指标，ICL 多任务基准）：

- `render_prompts_mc(item, continuation_delimiter, fewshot_examples)`（L19-36）：jinja2 Template [第三方]，
  多选题——fewshot 后接 `query + delimiter + 每个选项`，每个选项一个 prompt → `prompts[C]`。
- `render_prompts_schema(...)`（L39-57）：schema 题——fewshot 后每个 context 选项 + **同一个**
  continuation → `prompts[C]`。
- `render_prompts_lm(...)`（L60-92）：语言建模题——返回**两个** prompt（无/有 continuation；
  `|trim` 去尾空格 + `strip()` 保证 token 前缀干净）。
- `find_common_length(token_sequences, direction)`（L95-115）：多序列公共**前缀**（'left'）或
  **后缀**（'right'）长度。
- `stack_sequences(tokens, pad_token_id)`（L118-130）：不等长序列 →
  `torch.full((bsz, max_len), pad)` **右侧 BOS 填充** → `(C, T_pad) int64`。
- `batch_sequences_mc`（L133-149）：`tokenizer(prompts, prepend=bos)`（批量，↩ §2.1.3-⑥）→
  start = 公共前缀长度（所有选项相同），end = 各自长度。
- `batch_sequences_schema`（L152-168）：start = end − 公共后缀长度。
- `batch_sequences_lm`（L171-189）：断言 prompt_without 是 prompt_with 的前缀 →
  `[tokens_with], [start], [end]`（batch=1）。
- `forward_model(model, input_ids)`（L192-214）：

```
outputs = model(input_ids)                       (B, T_pad, 16384)   [GPT.forward targets=None → logits]
target_ids = torch.roll(input_ids, -1, 1)        (B, T_pad)          自回归目标 = 左移 1
losses = CE(outputs.view(B·T, V), target_ids.view(B·T), reduction='none').view(B, T_pad)   (B,T_pad)
losses[:, -1] = nan                              # 末列无目标
predictions = outputs.argmax(-1)                 (B, T_pad)
```

- `evaluate_example(idx, model, tokenizer, data, device, task_meta)`（L218-306）：
  fewshot 用 `random.Random(1234+idx)` 抽样（排除当前题，≤10 个）→ 按 task_type 渲染+批处理 →
  HF 模型 `max_seq_len` 截断（保留末 max_tokens、索引同步下移）→ `stack_sequences` → device →
  `forward_model` → 判定：
  - MC/schema：`mean_losses[i] = losses[i, si-1:ei-1].mean()`，取 **loss 最低的选项** == `item['gold']`；
  - LM：`predictions[0, si-1:ei-1] == input_ids[0, si:ei]` **全等**。
- `evaluate_task(model, tokenizer, data, device, task_meta)`（L309-337）：样本按
  `rank, rank+world_size, ...` 分派；`correct = zeros(len(data), fp32)`；DDP 时 barrier+all_reduce SUM
  （launch world=1 直算）；返回 `correct.mean()`。

CORE 形状全流程（launch：22 任务 × 20 题）：

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
```

##### ③-7 CSV 与报告收尾（L342-378，本文件）

`ddp_rank==0`：`base_eval/{model_slug}.csv`（`model_slug = f"base_model_{meta['step']:06d}"` →
**base_model_000100.csv**），逐任务写 `Task/Accuracy/Centered` + 最后一行 `CORE`；格式是 f-string
定宽 + CSV 分隔。`get_report().log(section="Base model evaluation / 基座模型评估", data=...)`
↩ §2.1.3-⑧（model 名、CORE、train/val bpb、样本）；`compute_cleanup()` ↩ §2.3.3-16。

#### 2.4.4 形状汇总（三种模式）

| 模式 | 形状要点 |
|---|---|
| sample | 条件 `(1,T_prompt)`→logits `(1,T,16384)`；无条件 num_samples=8 → decode KV `(2,8,T+128,1,128)` |
| bpb | x,y `(4,512)` → loss2d `(4,512)` → BPB 标量；4 步/split |
| core | 每题 input_ids `(C, T_pad)` → logits `(C,T_pad,16384)` → losses/predictions `(C,T_pad)` |

#### 2.4.5 产物与实测

`data/base_eval/base_model_000100.csv`（本机已存在 ✓）+ `data/report/base-model-evaluation---基座模型评估.md`。

---

### 2.5 `4a. SFT Mini` → `scripts/chat_sft_mini.py`

#### 2.5.1 定位与 CLI

```jsonc
"args": [
    "--device-batch-size", "4",     // 4 条 256 长序列/步
    "--num-iterations", "200",      // 200 步
    "--max-seq-len", "256",
    "--lr", "1e-4",                 // 所有参数组统一基准 lr（§2.5.3-⑤）
    "--run", "dummy",
    "--eval-every", "100"           // ⚠ 本脚本解析了但训练循环**未使用**（无 eval 逻辑，见 §6.2）
]
```

离线 SFT：只用本地 `data/identity_conversations.jsonl`（1000 条身份对话，已实测 1000 行），
不依赖 HuggingFace。加载 2a 产出的 d2 第 100 步模型 → 微调 → 存第 200 步检查点。
脚本无 `main()`，模块级顺序执行。

#### 2.5.2 调用树总图

```
scripts/chat_sft_mini.py（模块级顺序执行）
│
├─ [1] preflight_compile_check()          ↩ §2.3.2 ； autodetect_device_type / compute_init  ↩ §2.3.3-01
├─ [2] load_model("base", device, phase="train")   ← nanochat/checkpoint_manager.py ↩ 完整展开见 §2.4.3-③
│        （model_tag/step=None → find_largest_model/find_last_step → d2 / step 100）
├─ [3] CustomJSON(identity_path)          ← tasks/customjson.py，③-3 完整展开（首次使用）
│        └─ Task 基类（tasks/common.py，③-3 完整展开）
├─ [4] tokenizer.get_bos_token_id()       ← nanochat/tokenizer.py，③-4 完整展开
├─ [5] model.setup_optimizer(unembedding_lr=1e-4, embedding_lr=1e-4, matrix_lr=1e-4, weight_decay=0.0)
│        ↩ 完整展开见 §2.3.3-08（SFT 专属分组表见 ③-5）
├─ [6] data_generator()（本文件，③-7）
│        └─ tokenizer.render_conversation(conv)   ← nanochat/tokenizer.py，③-6 完整展开（首次使用）
│             └─ self.encode(content) / encode_special(...) / get_bos_token_id()   ↩ §2.1.3-⑥
├─ [7] sft_loader()（本文件，③-8）→ x,y (4,256)
├─ [8] 训练循环（本文件，③-9）: model(x, y) ↩ §2.3.3-15 → backward → optimizer.step ↩ §2.3.3-08
└─ [9] save_checkpoint(chatsft_checkpoints/d2, 200, model.state_dict(), None, meta)  ↩ §2.3.3-14
```

#### 2.5.3 逐方法展开

##### ③-1 启动与模型加载（L16-38，本文件）

`preflight_compile_check()` ↩ 完整展开见 §2.3.2（本脚本无 --no-compile → 走自动探测；launch 已注入
`TORCH_COMPILE_DISABLE=1`，且 import 顺序上本脚本不 import gpt/optim 装饰器路径——
checkpoint_manager 才间接 import gpt，时机安全）。
`compute_init` ↩ §2.3.3-01 → `(False,0,0,1,cpu)`。
`load_model("base", device, phase="train")` ↩ 完整展开见 §2.4.3-③。差异：**phase="train"** →
`model.train()`；`model_tag/step` 均 None → `find_largest_model`（d2）+ `find_last_step`（100）。

##### ③-2 数据断言（L40-46，本文件）

`get_base_dir()` ↩ §2.1.3-④ → `identity_path = data/identity_conversations.jsonl`；
`assert os.path.exists`，缺失时报错并提示 curl 下载（karpathy-public S3）。
`train_dataset = CustomJSON(filepath=identity_path)`（③-3），打印 `len(train_dataset)`=1000。

##### ③-3 `CustomJSON` + `Task` 基类（tasks/customjson.py + tasks/common.py）—— 完整展开（首次使用）

- `Task`（tasks/common.py L16-57）：对话数据集抽象基类。`__init__(start=0, stop=None, step=1)`
  轻量切片视图（断言 start≥0、stop≥start、step≥1）；`eval_type`（property）、`num_examples`、
  `get_example`、`evaluate` 留 NotImplemented；**`__len__`** = `ceil((stop−start)/step)`；
  **`__getitem__(index)`** = `get_example(start + index×step)`——`train_dataset[i]` 由此而来。
- `TaskMixture(Task)`（L60-92）：多任务混合——重复传同一任务即过采样；构造 `(task_idx, local_idx)`
  index_map 并用 `random.Random(42)` 打散。`TaskSequence(Task)`（L95-115）：顺序训练（curriculum）。
  `render_mc`（L118-137）：多选题渲染（字母放在选项**后**、delimiter 后无空格——token 绑定细节，
  小模型敏感）。三者 **launch 链路未用**（完整 SFT chat_sft.py 用），标注即可。
- `CustomJSON(Task)`（customjson.py L10-64）—— 完整展开：

```
__init__(filepath): 逐行读 JSONL
    文件缺失 → 打印 curl 提示（不 crash，length=0）
    每行: json.loads → 断言是 list、≥2 条消息、role 严格交替 user→assistant→...、
          content 必须是 str
num_examples() → self.length           # 实测 1000
get_example(index) → {"messages": messages}   # 包成 render_conversation 期望的字典格式
```

##### ③-4 `get_bos_token_id()`（tokenizer.py L278-280）—— 完整展开

`return self.bos_token_id` —— 训练/加载时由 `__init__` 存的实例变量（RustBPETokenizer 路径下 =
`encode_special("<|bos|>")` = **0**，↩ §2.1.3-③）。SFT 里用作打包填充 token（③-7）。
（HuggingFaceTokenizer 的同名方法 L153-165 有 `<|bos|>`→`<|endoftext|>` 的兜底查找，launch 不用。）

##### ③-5 优化器（脚本 L49）—— 回顾 + SFT 专属分组表

`model.setup_optimizer(unembedding_lr=1e-4, embedding_lr=1e-4, matrix_lr=1e-4, weight_decay=0.0)`
↩ 完整展开见 §2.3.3-08。与 2a 的差异：

| 差异点 | 2a（预训练） | 4a（SFT） |
|---|---|---|
| AdamW 组 lr | unembedding 0.008 / embedding 0.3 × batch_scale | 全部基准 1e-4 |
| dmodel 缩放 | √6 照常 | √6 照常 → 嵌入类 ≈ **2.45e-4** |
| Muon 组 lr | 0.02×0.25=0.005 | **1e-4**（内部再按 √(m/n)：(512,128)→2e-4） |
| Muon wd | ≈2.74（余弦衰减） | **0**（AdamW 组固定 wd 不受此参数影响） |
| `scalar_lr` | 0.5×0.25 | 未传 → 默认 **0.5**（x0 组 lr=0.5，偏大——作者取舍，未覆盖） |
| `initial_lr` 记录 | 有 | 有（本脚本无调度器 → 恒 1.0） |

##### ③-6 `render_conversation(conversation, max_tokens=2048)`（tokenizer.py L330-430）—— 完整展开（首次使用）

SFT 最关键的预处理：一条对话 → `(ids, mask)`。**mask=1 的只有 assistant 输出 token（参与损失）**。

```
system 消息 → 合并进第一条 user（深拷贝避免改原数据）
断言消息 ≥1 条且角色严格交替 user→assistant→...
取特殊 token id: bos/user_start/user_end/assistant_start/assistant_end/python_*/output_*（encode_special ↩ §2.1.3-⑥）
add_tokens(ids, mask_val) 闭包: 批量 extend ids 与等长 mask
[bos, 0]
user 消息:  [user_start,0] + [encode(content),0] + [user_end,0]        # content 必须是 str
assistant 消息: [assistant_start,0] + 按 content 分:
     str          → [encode(content), 1]                               # 纯文本回复 → 全监督
     list(parts)  → part.type=="text"         → [encode(text), 1]
                    part.type=="python"       → [python_start,1]+[tokens,1]+[python_end,1]   # 工具调用也要学
                    part.type=="python_output"→ [output_start,0]+[tokens,0]+[output_end,0]   # 工具输出不监督
              + [assistant_end, 1]
ids/mask 截断到 max_tokens=2048
```

```
 [<|bos|>|<|user_start|> 你好 <|user_end|>|<|assistant_start|> 回复 <|assistant_end|> ...]
 mask   0         0       0         0          0               1       1

 ← 只有 assistant 输出的 token 参与损失（mask=1）
```

`render_for_completion`（L450-472，RL 用：弹掉最后一条 assistant 消息再补 `<|assistant_start|>` 引导生成）与
`visualize_tokenization`（L432-448，终端彩色调试）**launch 未用**，标注即可。

##### ③-7 `data_generator()`（L55-86，本文件）—— 完整展开

把对话拼成固定长 257 的行（row_capacity = max_seq_len+1）：

```
conv_buffer = []; idx = 0
while True:
    for _ in range(4):                       # 每轮 4 行（device_batch_size）
        row, mask_row = [], []
        while len(row) < 257:
            if not conv_buffer:              # 预渲染 32 条对话（循环取模 → 无限轮换）
                conv_buffer = [render_conversation(train_dataset[(idx+i) % 1000]) for i in range(32)]
                idx += 32
            ids, mask = conv_buffer.pop(0)
            if len(ids) <= 257 - len(row):   # 放得下 → 整条放入
                row.extend(ids);  mask_row.extend(mask)
            else:                            # 放不下 → 放回缓冲，用 <|bos|>(mask=0) 填满剩余
                conv_buffer.insert(0, (ids, mask))
                row.extend([0] × (257 - len(row)));  mask_row.extend([0] × ...)
                break
        yield row[:257], mask_row[:257]      # 截断保护
```

```
 [<|bos|>|...对话1...|<|bos|>|...对话2(截断至行满)...|<|bos|><|bos|>...填充]
 mask  0      ...        0       1                   1        0     0   ...
```

##### ③-8 `sft_loader()`（L88-116，本文件）—— 完整展开

```
rows, mask_rows = [], [];  gen = data_generator()
while True:
    for _ in range(4):  r, m = next(gen);  rows.append(r);  mask_rows.append(m)
    batch   = torch.tensor(rows, dtype=torch.long)      (4, 257) int64
    inputs  = batch[:, :-1].to(device)                  (4, 256) int64 → device
    targets = batch[:, 1:].to(device).clone()           (4, 256) int64 → device
    mask    = torch.tensor(mask_rows, dtype=torch.int8) (4, 257) int8
    targets[mask[:, 1:] == 0] = -1                      (4, 256) ← 非 assistant 位置不监督
    rows.clear(); mask_rows.clear()
    yield inputs, targets
```

##### ③-9 训练循环（L118-146，本文件）—— 完整展开

```
smooth_loss = 0; step = 0
while step < 200:
    model.train()
    loss = model(x, y)            # GPT.forward ↩ 完整展开见 §2.3.3-15；B=4,T=256
        x (4,256) → ... → logits (4,256,16384)
        loss = CE(logits.view(1024, 16384), y.view(1024), ignore_index=-1)   # 只对 mask=1 位置求梯度
    loss.backward()               # 无梯度累积（device_batch × seq_len 即全局批）
    optimizer.step()              # MuonAdamW.step ↩ §2.3.3-08（无调度器，lr 恒为 initial_lr）
    model.zero_grad(set_to_none=True)
    x, y = next(loader)
    smooth_loss = 0.9·smooth_loss + 0.1·loss.item()
    step%10 打印（去偏 /(1−0.9^step)）
# 保存
save_checkpoint(data/chatsft_checkpoints/d2, 200, model.state_dict(), None,
                {"step": 200, "model_config": meta["model_config"]})
```

`save_checkpoint` ↩ 完整展开见 §2.3.3-14。差异：`optimizer_data=None` → **不写 optim 分片**（SFT 不可续训）；
meta 只存 `step + model_config`。

#### 2.5.4 形状全流程（B=4, T=256）

```
data/identity_conversations.jsonl（实测 1000 行）
  → CustomJSON 逐行校验（角色交替 user→assistant, ≥2 条消息）
  → train_dataset[i] = {"messages": [...]}
  → tokenizer.render_conversation(conv) → (ids, mask)
  → data_generator: 每轮预渲染 32 条 → 逐条拼进 row(257)，塞不下 → <|bos|>(mask=0) 填满
  → sft_loader:
      batch = tensor(rows)         (4, 257) int64
      x = batch[:, :-1]            (4, 256) int64 → device
      y = batch[:, 1:].clone()     (4, 256) int64 → device
      mask = tensor(mask_rows)     (4, 257) int8  → 用 mask[:, 1:] (4, 256) 对齐 y
      y[mask[:, 1:] == 0] = -1    ← 非 assistant token 不监督
      yield x, y
前向（T=256，与 §2.3.3-15 相同结构，B=4）:
  x (4,256) → ... → logits (4,256,16384)
  loss = CE(logits.view(1024,16384), y.view(1024), ignore_index=-1)
训练: 200 步，无梯度累积；step%10 打印 EMA loss；无 eval（--eval-every 100 未被使用）
保存: data/chatsft_checkpoints/d2/model_000200.pt + meta_000200.json
```

#### 2.5.5 产物与实测

| 产物 | 说明 | 实测 |
|---|---|---|
| `data/chatsft_checkpoints/d2/model_000200.pt` | SFT 后模型权重 | 存在 ✓ |
| `data/chatsft_checkpoints/d2/meta_000200.json` | step/model_config | 存在 ✓ |
| `data/identity_conversations.jsonl` | SFT 数据（需 curl 下载） | 1000 行 ✓ |

### 2.6 `6a/6b. Chat CLI` → `scripts/chat_cli.py`

#### 2.6.1 定位与 CLI

| 差异 | `6a. Chat CLI` | `6b. Chat CLI 单次问答` |
|---|---|---|
| `--prompt` | 无 | `"你好，请自我介绍一下。"` |
| `--top-k` | 50（显式） | 未传 → 默认 50 |
| 行为 | 循环读 stdin 直到 quit/exit | 单轮回答后 `break` 退出 |

共同参数：`--source sft`（→ `chatsft_checkpoints/`）、`--model-tag d2`、`--step 200`（4a 的产物）、
`--temperature 0.6`、`--device-type cpu`。脚本无 `main()`：初始化后进入 `while True` 交互循环。

#### 2.6.2 调用树总图

```
scripts/chat_cli.py（模块级 + while True 交互循环）
│
├─ [1] argparse（-i/--source, -g/--model-tag, -s/--step, -p/--prompt, -t/--temperature, -k/--top-k, --device-type）
├─ [2] autodetect_device_type / compute_init          ↩ §2.3.3-01
├─ [3] load_model(args.source, device, phase="eval", model_tag="d2", step=200)
│      ← nanochat/checkpoint_manager.py ↩ 完整展开见 §2.4.3-③（source="sft" → data/chatsft_checkpoints）
├─ [4] tokenizer.get_bos_token_id()                   ↩ §2.5.3-④
├─ [5] tokenizer.encode_special("<|user_start|>") 等 ×4   ↩ §2.1.3-⑥
├─ [6] Engine(model, tokenizer)                       ↩ §2.3.3-13
└─ [7] while True:
     ├─ user_input = args.prompt 或 input("\nUser: ")
     ├─ conversation_tokens += [user_start] + tokenizer.encode(user_input) + [user_end] + [assistant_start]
     │                        （encode ↩ §2.1.3-⑥）
     ├─ for token_column, token_masks in engine.generate(conversation_tokens, num_samples=1,
     │        max_tokens=256, temperature=0.6, top_k=50):          ← ③-4 对话版大图
     │        token = token_column[0]; tokenizer.decode([token]) 打印   （decode ↩ §2.1.3-⑥）
     └─ response_tokens 补 assistant_end → 拼回 conversation_tokens；--prompt 模式 break
```

#### 2.6.3 逐方法展开

##### ③-1 启动初始化（L28-30，本文件）

`autodetect_device_type` / `compute_init` ↩ §2.3.3-01 → `(False,0,0,1,cpu)`。

##### ③-2 模型加载（L30，本文件）

`load_model(args.source, device, phase="eval", model_tag="d2", step=200)` ↩ 完整展开见 §2.4.3-③。
差异：`source="sft"` → 目录 **`data/chatsft_checkpoints`**（load_model 的 source 映射表）；
`phase="eval"` → `model.eval()`；`step=200` → 4a 的 `model_000200.pt`。返回
`(model, tokenizer, meta_000200)`。

##### ③-3 聊天状态机的特殊 token（L34-36，本文件）

```
bos = tokenizer.get_bos_token_id()                              ↩ §2.5.3-④ → 0
user_start, user_end             = encode_special("<|user_start|>"),  encode_special("<|user_end|>")
assistant_start, assistant_end   = encode_special("<|assistant_start|>"), encode_special("<|assistant_end|>")
```

`encode_special` ↩ 完整展开见 §2.1.3-⑥ → id 16376/16377/16378/16379（§2.1.3-③ 的 SPECIAL_TOKENS 表）。

##### ③-4 对话生成 —— `Engine.generate` 回顾 + 对话版 prefill/decode 大图

`Engine` 全体 ↩ 完整展开见 §2.3.3-13。本处为 chat 场景的完整形状流：

```
engine.generate(conversation_tokens, num_samples=1, max_tokens=256, temperature=0.6, top_k=50)

┌─ ① Prefill（一次前向，B=1）────────────────────────────┐
│ kv_cache_prefill = KVCache(B=1, seq_len=len(tokens),     │
│                             2层, 4头, D=32, dtype=fp32)  │
│   k_cache/v_cache  (2, 1, len(tokens), 4, 32)            │
│ ids = tensor([[tokens]])              (1, T_conv)        │
│ logits = model(ids, kv_cache)         (1, T_conv, 16384) │
│   └ 每层: q (1,T_conv,4,32) ; KV 写入缓存 ;              │
│     Smear 在 prefill(T>1) 用训练同款路径（§2.3.3-15 ③）; │
│     每层结束 kv_cache.advance? 否——最后一层处理后推进    │
│ logits[:,-1,:] → (1, 16384) → expand(1,-1)               │
├─ ② 广播 ───────────────────────────────────────────────┤
│ kv_cache_decode = KVCache(B=1, seq_len=T_conv+256, ...)  │
│   k_cache/v_cache  (2, 1, T_conv+256, 4, 32)            │
│ prefill() 拷贝 KV 与 prev_embedding                      │
├─ ③ Decode 循环（每步）─────────────────────────────────┤
│ ids (1,1) → forward(kv_cache=decode)                     │
│   RoPE 偏移 T0=pos ; Smear 用 prev_embedding (1,1,128)   │
│   q (1,1,4,32); k,v (1,1,4,32) 写入缓存 pos             │
│   SDPA: k_full (1,pos+1,4,32) → y (1,1,4,32) →view(1,1,128)│
│   → logits (1,1,16384) → sample_next_token(0.6, top_k=50)│
│   → (1,1) → yield (token_column, token_masks)            │
└─────────────────────────────────────────────────────────┘
终止: <|assistant_end|> 或 <|bos|>（行完成标记）或 max_tokens=256
```

##### ③-5 对话循环（L48-113，本文件）—— 完整展开

```
conversation_tokens = [bos]
while True:
    user_input = args.prompt（6b: 一次性）或 input("\nUser: ").strip()（6a: 交互）
    quit/exit → break；clear → conversation_tokens=[bos] 重开；空输入 → continue
    拼用户消息: conversation_tokens += [user_start] + encode(user_input) + [user_end] + [assistant_start]
    response_tokens = []
    for token_column, token_masks in engine.generate(conversation_tokens, num_samples=1,
                                                     max_tokens=256, temperature=0.6, top_k=50):
        token = token_column[0]                 # 去掉 batch 维（num_samples=1）
        response_tokens.append(token)
        tokenizer.decode([token]) → 逐 token 流式打印（flush=True）
    若 response_tokens[-1] != assistant_end: append(assistant_end)   # 保证对话格式完整
    conversation_tokens.extend(response_tokens)
    6b（--prompt）→ break
```

`tokenizer.encode/decode` ↩ 完整展开见 §2.1.3-⑥（单串路径 → `enc.encode_ordinary`；decode 保留特殊 token）。

#### 2.6.4 推理形状总表（num_samples=1）

| 阶段 | 张量 | 形状 |
|---|---|---|
| Prefill 输入 | `ids` | (1, T_conv) |
| Prefill KV | `k_cache/v_cache` | (2, 1, T_conv, 4, 32) |
| Prefill logits | `logits` | (1, T_conv, 16384) → 取尾 (1, 16384) |
| Decode 缓存 | `k_cache/v_cache` | (2, 1, T_conv+256, 4, 32) |
| Decode 每步输入 | `ids` | (1, 1) |
| Decode 每步 | `q/k/v` | (1, 1, 4, 32)；`logits` (1, 1, 16384) → 采样 (1, 1) |
| Smear | `prev_embedding` | (1, 1, 128) |

#### 2.6.5 产物与实测

无落盘产物——纯终端输出（6a 交互对话；6b 单次回答）。

---

## 3. 全局依赖树（递归刨根问到底）

把 6 个入口脚本 `import` 的项目内模块**递归展开到叶子**。项目内模块追到底（细节见 §2 各执行小节）；
第三方库（venv 里的包）标注职责后停住——那是"根"，没有再往下的项目代码。

```
.vscode/launch.json
│
├─ 1b → scripts/tok_train.py                    （逐方法展开 §2.1）
│     ├─ nanochat/tokenizer.py                 （train_from_iterator/save/encode/decode，§2.1.3-③⑤⑥）
│     │     ├─ rustbpe                 [第三方] Rust 实现的 BPE 训练器
│     │     ├─ tiktoken               [第三方] OpenAI BPE 推理（encode/decode）
│     │     ├─ tokenizers             [第三方] HuggingFace Tokenizer（HuggingFaceTokenizer 用，launch 未用）
│     │     └─ (pickle)               序列化 Encoding 到 tokenizer.pkl
│     ├─ nanochat/common.py           (get_base_dir，§2.1.3-④)
│     │     ├─ torch / torch.distributed   [第三方] DDP（未激活，无 RANK 环境变量）
│     │     └─ filelock               [第三方] 下载文件锁
│     ├─ nanochat/dataset.py          (parquets_iter_batched，§2.1.3-②)
│     │     ├─ requests               [第三方] HTTP 下载 shard
│     │     └─ pyarrow.parquet        [第三方] 列式读取 parquet
│     └─ nanochat/report.py           (get_report().log，§2.1.3-⑧)
│           ├─ psutil / socket / platform / subprocess   [第三方] 系统信息与 git 命令
│           └─ nanochat/common.py     (get_base_dir / get_dist_info)
│
├─ 1c → scripts/tok_eval.py                    （逐方法展开 §2.2）
│     ├─ nanochat/tokenizer.py         （from_pretrained/get_tokenizer，§2.2.3-②③）
│     ├─ nanochat/dataset.py           （↩ §2.1.3-②）
│     └─ nanochat/report.py            （↩ §2.1.3-⑧）
│
├─ 2a → scripts/base_train.py                  （逐方法展开 §2.3）
│     ├─ nanochat/common.py            (preflight_compile_check 在 import gpt 前执行！§2.3.2；设备家族 §2.3.3-01)
│     ├─ nanochat/gpt.py               ← 模型本体（§2.3.3-04/05/06/15）
│     │     ├─ nanochat/common.py      (get_dist_info / COMPUTE_DTYPE)
│     │     ├─ nanochat/optim.py       (MuonAdamW / DistMuonAdamW，§2.3.3-08)
│     │     │     └─ torch.compile     (条件装饰器：TORCH_COMPILE_DISABLE=1 时 no-op)
│     │     └─ nanochat/flash_attention.py   (§2.3.3-15)
│     │           ├─ kernels [第三方]  FA3 kernel（仅 CUDA SM90；CPU → None → SDPA 回退）
│     │           └─ torch.nn.functional.scaled_dot_product_attention   (SDPA 回退实现)
│     ├─ nanochat/dataloader.py        (§2.3.3-09)
│     │     ├─ nanochat/common.py, nanochat/dataset.py（↩ §2.1.3-②）
│     │     └─ pyarrow.parquet
│     ├─ nanochat/tokenizer.py         (get_tokenizer/get_token_bytes，§2.2.3-③ / §2.3.3-03)
│     ├─ nanochat/checkpoint_manager.py（save_checkpoint §2.3.3-14；resume 死分支 load_checkpoint ↩ §2.4.3-③）
│     │     ├─ nanochat/common.py
│     │     ├─ nanochat/gpt.py         (GPT/GPTConfig → 重建模型)
│     │     └─ nanochat/tokenizer.py
│     ├─ nanochat/loss_eval.py         (evaluate_bpb，§2.3.3-12)
│     ├─ nanochat/engine.py            (采样用，§2.3.3-13)
│     │     └─ nanochat/checkpoint_manager.py（仅 __main__ 测试用）
│     ├─ scripts/base_eval.py          (evaluate_core —— 跨脚本导入！2a 里是死分支，真正执行 §2.4.3-⑥)
│     │     ├─ nanochat/common.py / tokenizer.py / checkpoint_manager.py / dataloader.py / loss_eval.py / engine.py
│     │     ├─ nanochat/core_eval.py  (§2.4.3-⑥)
│     │     │     └─ jinja2 [第三方]  prompt 模板渲染
│     │     ├─ transformers [第三方]   仅 --hf-path 时加载 HF 模型（launch 未用）
│     │     └─ yaml / csv / zipfile
│     ├─ nanochat/fp8.py               [仅 --fp8 + CUDA 时 import；CPU launch 是死分支]
│     ├─ nanochat/report.py            (↩ §2.1.3-⑧)
│     └─ wandb [第三方]                (--run dummy → DummyWandb，不联网)
│
├─ 3a → scripts/base_eval.py                   （逐方法展开 §2.4）
│     ├─ nanochat/common.py / tokenizer.py / checkpoint_manager.py（§2.4.3-③）
│     ├─ nanochat/core_eval.py         (jinja2，§2.4.3-⑥)
│     ├─ nanochat/dataloader.py / loss_eval.py / engine.py（↩ §2.3.3-09/12/13）
│     ├─ nanochat/report.py
│     └─ yaml / csv / zipfile / tempfile / shutil / random
│
├─ 4a → scripts/chat_sft_mini.py               （逐方法展开 §2.5）
│     ├─ nanochat/common.py            (compute_init / preflight_compile_check)
│     ├─ nanochat/checkpoint_manager.py   (load_model / save_checkpoint)
│     ├─ nanochat/tokenizer.py         (get_tokenizer / render_conversation / get_bos_token_id)
│     ├─ tasks/customjson.py           (§2.5.3-③)
│     │     └─ tasks/common.py         (Task 基类)
│     └─ (间接) nanochat/gpt.py → optim.py → flash_attention.py
│
├─ 6a/6b → scripts/chat_cli.py                 （逐方法展开 §2.6）
│     ├─ nanochat/common.py            (compute_init / autodetect_device_type)
│     ├─ nanochat/engine.py            (Engine + KVCache，§2.3.3-13 / §2.6.3-④)
│     │     └─ nanochat/checkpoint_manager.py
│     └─ nanochat/checkpoint_manager.py   (load_model)
│
└─ Python: 调试当前文件   （${file} 通用配置，不追踪）
```

**第三方依赖职责一句话（刨根到此为止）**：

| 库 | 职责 | 谁在用 |
|---|---|---|
| `rustbpe` | Rust 实现的高性能 BPE **训练** | tok_train（§2.1.3-③） |
| `tiktoken` | OpenAI BPE **推理**（encode/decode/batch） | tokenizer（全项目） |
| `tokenizers` | HuggingFace 分词器实现 | tokenizer 的 HuggingFaceTokenizer 类（launch 未用） |
| `pyarrow` | parquet 列式读（row_group 粒度） | dataset / dataloader（§2.1.3-②/§2.3.3-09） |
| `requests` | HTTP 流式下载数据分片 | dataset |
| `filelock` | 多进程下载互斥锁 | common（§2.4.3-⑥） |
| `jinja2` | CORE prompt 模板 | core_eval（§2.4.3-⑥） |
| `wandb` | 云端训练日志（`--run dummy` 时不激活） | base_train |
| `psutil`/`socket`/`platform` | 报告卡系统信息 | report（§2.1.3-⑧） |
| `transformers` | 加载 HuggingFace 模型（仅 `--hf-path`） | base_eval（launch 未用，§2.4.3-②） |
| `kernels` | FA3 内核包（仅 CUDA SM90；CPU 返回 None） | flash_attention（§2.3.3-15） |

---

## 4. 模块 × 方法索引（反向速查）

> 阅读某个"文件"的全部方法时，按此表找到完整展开位置。回顾处给的是调用上下文与形状差异。

| 模块 | 方法 | 完整展开 | 回顾 / 备注 |
|---|---|---|---|
| `nanochat/common.py` | `get_base_dir` | §2.1.3-④ | §2.3/2.4/2.5 |
| | `autodetect_device_type / get_dist_info / is_ddp_requested / is_ddp_initialized / compute_init / COMPUTE_DTYPE / print0 / get_peak_flops / DummyWandb / ColoredFormatter+setup_default_logging` | §2.3.3-01 | §2.4/2.5/2.6 |
| | `preflight_compile_check` | §2.3.2 | §2.5.3-① |
| | `download_file_with_lock` | §2.4.3-⑥ | — |
| | `compute_cleanup` | §2.3.3-16 | §2.4 |
| | `print_banner` | §2.3.2 | — |
| `nanochat/tokenizer.py` | `SPECIAL_TOKENS / SPLIT_PATTERN` | §2.1.3-③ | — |
| | `RustBPETokenizer.__init__ / train_from_iterator` | §2.1.3-③ | — |
| | `save` | §2.1.3-⑤ | — |
| | `encode / decode / encode_special / __call__ / id_to_token` | §2.1.3-⑥ | §2.2/2.3/2.5/2.6 |
| | `get_vocab_size / get_special_tokens` | §2.1.3-⑦ | §2.3/2.4 |
| | `from_pretrained` | §2.2.3-② | — |
| | `from_directory / get_tokenizer` | §2.2.3-③ | §2.3/2.5 |
| | `get_token_bytes` | §2.3.3-03 | §2.4 |
| | `get_bos_token_id` | §2.5.3-④ | §2.6 |
| | `render_conversation` | §2.5.3-⑥ | — |
| | `render_for_completion / visualize_tokenization` | §2.5.3-⑥ 注 | launch 未用 |
| | `HuggingFaceTokenizer` 整类 | §2.4.3-② | launch 未用（死分支） |
| `nanochat/dataset.py` | `list_parquet_files / parquets_iter_batched` | §2.1.3-② | §2.2/2.3 |
| | `download_single_file / __main__` | §2.1.3-② 注 | launch 未用 |
| `nanochat/dataloader.py` | `_document_batches / refill_buffer / tokenizing_distributed_data_loader_with_state_bos_bestfit / ..._bos_bestfit` | §2.3.3-09 | §2.4 |
| `nanochat/gpt.py` | `GPTConfig` | §2.3.3-04 | §2.4 |
| | `norm / Linear / has_ve` | §2.3.3-05 | — |
| | `GPT.__init__ / _compute_window_sizes / _precompute_rotary_embeddings` | §2.3.3-05 | — |
| | `init_weights` | §2.3.3-06 | §2.4.3-③ |
| | `get_device` | §2.3.3-05 | — |
| | `num_scaling_params / estimate_flops` | §2.3.3-07 | — |
| | `setup_optimizer` | §2.3.3-08 | §2.5 |
| | `CausalSelfAttention / MLP / Block`（结构详析）/ `apply_rotary_emb` | §2.3.3-05 | 前向形状流见 §2.3.3-15 |
| | `GPT.forward` | §2.3.3-15 | §2.4/2.5/2.6 |
| | `GPT.generate` | §2.3.3-15 注 | launch 未用（engine 自测用） |
| `nanochat/optim.py` | `_conditional_compile / adamw_step_fused / muon_step_fused` | §2.3.3-08 | — |
| | `MuonAdamW（__init__ / _step_adamw / _step_muon / step）` | §2.3.3-08 | §2.5 |
| | `DistMuonAdamW` | §2.3.3-08 注 | launch 不激活 |
| `nanochat/flash_attention.py` | `_load_flash_attention_3 / HAS_FA3 / USE_FA3 / _sdpa_attention / flash_attn_func / flash_attn_with_kvcache` | §2.3.3-15 | §2.6 |
| `nanochat/loss_eval.py` | `evaluate_bpb` | §2.3.3-12 | §2.4 |
| `nanochat/checkpoint_manager.py` | `save_checkpoint` | §2.3.3-14 | §2.5 |
| | `load_checkpoint / build_model / find_largest_model / find_last_step / load_model_from_dir / load_model / _patch_missing_config_keys / _patch_missing_keys` | §2.4.3-③ | §2.5/2.6 |
| | `load_optimizer_state / log0` | §2.4.3-③ 注 | launch 未用 |
| `nanochat/engine.py` | `timeout / eval_with_timeout / use_calculator / KVCache(全方法) / sample_next_token / RowState / Engine.generate / Engine.generate_batch` | §2.3.3-13 | §2.4/2.6 |
| `nanochat/core_eval.py` | `render_prompts_mc/schema/lm / find_common_length / stack_sequences / batch_sequences_mc/schema/lm / forward_model / evaluate_example / evaluate_task` | §2.4.3-⑥ | — |
| `nanochat/report.py` | `get_report / Report.log` | §2.1.3-⑧ | §2.2/2.3/2.4 |
| | `Report.generate / reset / generate_header 等辅助` | §2.1.3-⑧ 注 | launch 未用 |
| `scripts/base_eval.py` | `evaluate_core / place_eval_bundle` | §2.4.3-⑥ | §2.3（死分支） |
| | `ModelWrapper / load_hf_model / get_hf_token_bytes` | §2.4.3-② | launch 未用（死分支） |
| `tasks/common.py` | `Task / TaskMixture / TaskSequence / render_mc` | §2.5.3-③ | 后三者 launch 未用 |
| `tasks/customjson.py` | `CustomJSON` | §2.5.3-③ | — |
| 脚本内函数 | `text_iterator`（tok_train） | §2.1.3-① | — |
| | `print_comparison`（tok_eval） | §2.2.3-⑤ | — |
| | `build_model_meta / get_scaling_params`（base_train） | §2.3.3-04 / §2.3.3-07 | — |
| | `get_lr_multiplier / get_muon_momentum / get_weight_decay`（base_train） | §2.3.3-10 | — |
| | `disable_fp8`（base_train） | §2.3.3-06 注 | launch 下空转 |
| | `main`（base_eval） | §2.4.3-① | — |
| | `data_generator / sft_loader`（chat_sft_mini） | §2.5.3-⑦ / §2.5.3-⑧ | — |

---

## 5. 张量/矩阵形状速查表

> 各执行小节的形状图入口：前向七步大图 §2.3.3-15；梯度累积 §2.3.4；CORE §2.4.3-⑥；SFT §2.5.4；推理 §2.6.3-④。

### 5.1 参数总表（d2，launch 2a 实际配置）

`V=16384, C=128, 4C=512`。init 中 `s = √3/√128 ≈ 0.1531`（均匀分布边界，等 std）。初始化策略细节见 §2.3.3-06。

| 参数 | 形状 | 数量 | 占比 | 初始化 | 优化器组 |
|---|---|---|---:|---:|---|---|
| `wte.weight` | (16384, 128) | 2,097,152 | 31.4% | N(0, 0.8) | adamw-embedding |
| `lm_head.weight` | (16384, 128) | 2,097,152 | 31.4% | N(0, 0.001) | adamw-unembedding |
| `value_embeds."1".weight` | (16384, 128) | 2,097,152 | 31.4% | U(±s) | adamw-ve |
| 层0: `attn.c_q/c_k/c_v/c_proj` | (128,128) ×4 | 65,536 | 1.0% | U(±s) / c_proj=0 | muon |
| 层0: `mlp.c_fc` | (512,128) | 65,536 | 1.0% | U(±0.4s) | muon |
| 层0: `mlp.c_proj` | (128,512) | 65,536 | 1.0% | 0 | muon |
| 层1: 同上 6 个矩阵 | 同上 | 196,608 | 2.9% | 同上 | muon |
| 层1: `attn.ve_gate` | (4,12) | 48 | ~0 | U(0, 0.02) | muon |
| `resid_lambdas` | (2,) | 2 | ~0 | 1.15→1.05 | adamw |
| `x0_lambdas` | (2,) | 2 | ~0 | 0.20→0.05 | adamw |
| `smear_gate.weight` | (1,24) | 24 | ~0 | U(0, 0.02) | adamw |
| `smear_lambda` | (1,) | 1 | ~0 | 0 | adamw |
| `backout_lambda` | (1,) | 1 | ~0 | 0.2 | adamw |
| **合计** | — | **6,684,750** | 100% | — | — |

> 注意：层 0 无 VE（`has_ve(0,2)=False`），层 1 有（最后一层必开）。三大嵌入各占 31%，合计 94%，
> 这正是 launch 注释说 "~4M" 而实际 ~6.7M 的原因（注释可能按 tied 权重或忽略 VE 估算，见 §6.2）。
> 参数分组核算 `num_scaling_params` 见 §2.3.3-07；优化器 10 组实测表见 §2.3.3-08。

### 5.2 激活张量总表（前向，d2，训练 B=1/T=512）

| 阶段 | 张量 | 形状 | dtype |
|---|---|---|---|
| 输入 | `idx` / `y` | (1, 512) | int64 |
| 嵌入 | `x = wte(idx)` | (1, 512, 128) | fp32 |
| Smear | `gate` | (1, 511, 1) | fp32 |
| 逐层缩放 | `x0`, `resid·x + x0·x0` | (1, 512, 128) | fp32 |
| VE | `ve`（仅层1） | (1, 512, 128) | fp32 |
| VE 门 | `gate = 3σ(ve_gate(x[...,:12]))` | (1, 512, 4) | fp32 |
| Q/K/V | `q`, `k`, `v` | (1, 512, 4, 32) | fp32 |
| RoPE | `cos`, `sin` | (1, 512, 1, 16) | fp32 |
| 注意力 | `y`（SDPA 内部转 (1,4,512,32)） | (1, 512, 4, 32) | fp32 |
| 输出投影 | `attn_out / mlp_out` | (1, 512, 128) | fp32 |
| MLP 中间 | `c_fc(x)` | (1, 512, 512) | fp32 |
| Backout | `x_backout`（层1后） | (1, 512, 128) | fp32 |
| lm_head | `logits` | (1, 512, 16384) | fp32（softcap 后） |
| 损失 | `loss` | 标量 () | fp32 |

### 5.3 推理张量总表（6a/6b，num_samples=1）

| 阶段 | 张量 | 形状 |
|---|---|---|
| Prefill 输入 | `ids` | (1, T_conv) |
| Prefill KV | `k_cache/v_cache` | (2, 1, T_conv, 4, 32) |
| Prefill logits | `logits` | (1, T_conv, 16384) → 取尾 (1, 16384) |
| Decode 缓存 | `k_cache/v_cache` | (2, 1, T_conv+256, 4, 32) |
| Decode 每步输入 | `ids` | (1, 1) |
| Decode 每步 | `q/k/v` | (1, 1, 4, 32)；`logits` (1, 1, 16384) → 采样 (1, 1) |
| Smear | `prev_embedding` | (1, 1, 128) |

（KVCache 成员定义见 §2.3.3-13；对话版流程大图见 §2.6.3-④。）

### 5.4 通用公式（任意 depth d；head_dim 由 `--head-dim` 设置，默认 128）

```
model_dim  C = ceil(d × 64 / head_dim) × head_dim       n_head H = C / head_dim
wte / lm_head / value_embeds: (V, C)           每个 block 矩阵: 4×C² + 2×4C² = 6C²
总参数 ≈ 3·V·C + d·6C² + H·12·(VE 层数)       (V 大时嵌入占主导)
FLOPs/token ≈ 6·(d·6C² + V·C) + Σ_layers 12·H·head_dim·window_l     （推导 §2.3.3-07）
（H×head_dim = C 恒成立 → 注意力项只由 C 与窗口决定，与头的拆法无关）
```

---

## 6. 附录

### 6.1 实际运行产物（本机 data/ 目录，均已核实存在）

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

> 注：盘上 d2 检查点与指标（`val_bpb=1.9506` 等）为**改动前**（head_dim=128、1 头）运行产物；
> 切到方案 C（`--head-dim 32` → 4 头）重跑 2a 后自动更新（§2.3.5）。

### 6.2 与 launch.json 注释的偏差说明

| launch 注释说法 | 实际情况 |
|---|---|
| 2a 名称写 "~4M params" | 实测 **6,684,750 ≈ 6.7M**（§2.3.3-07）。untied 的 wte+lm_head+VE 三个 (16384,128) 嵌入各 2.1M，注释可能按 tied 估算 |
| 2a 注释 "depth=2, 极小模型" | ✓ 正确；n_embd=128 由 --depth 推导；n_head=4 由 `--head-dim 32` 推导（改动前默认 head_dim=128 → 1 头，§2.3.3-04） |
| window_pattern 未在 args 中指定 | 走默认 "SSSL" → 层0 窗口 128、层1 全上下文；SDPA 回退会打效率警告但功能正确（§2.3.3-05/15） |
| `--eval-every 100`（4a） | chat_sft_mini.py 解析了该参数但训练循环**并未使用**（无 eval 逻辑，§2.5.1） |
| 3a `--eval core,bpb,sample` | ✓ 三模式都执行（CORE 22 任务×20 题；BPB 每 split 4 步；7+8 采样，§2.4） |

### 6.3 与 launch 链路相关的环境/依赖前提

- 4a 依赖 `data/identity_conversations.jsonl`，脚本 assert 存在，否则提示 curl 下载（karpathy-public S3）。
- 3a 的 CORE 依赖 `data/eval_bundle/`，缺失时 `download_file_with_lock` 自动从 S3 下载 zip 并解压（§2.4.3-⑥）。
- 2a/4a 的数据加载依赖 `data/base_data_climbmix/*.parquet`，缺失时需 `python -m nanochat.dataset -n <N>` 下载。
- Windows 无 MSVC：2a 必须 `--no-compile`（且 launch 注入了 `TORCH_COMPILE_DISABLE=1`），
  否则 optim.py 里的 `@torch.compile` 装饰器会在 import 期报错（§2.3.2）。
- Windows 无 `signal.SIGALRM`：Engine 的 `use_calculator` 超时保护失效 → 工具调用计算器静默返回 None，
  不影响正常生成（§2.3.3-13）。

### 6.4 不在 launch 链路中的模块（仅标注引用关系，不展开）

- `nanochat/fp8.py`：FP8 混合精度训练。仅在 `base_train.py --fp8` **且 CUDA** 时 import（launch 2a 是 CPU → 死分支，§2.3.3-06）。
- `nanochat/execution.py`：仅被 `tasks/humaneval.py` 使用（chat_eval 链路，launch 未涉及）。
- `scripts/chat_sft.py / chat_rl.py / chat_eval.py / chat_web.py`：完整 SFT / RL / 对话评估 / Web 服务，launch 未涉及。
- `nanochat/report.py` 的 `generate()/reset()`：需手动 `python -m nanochat.report generate` 才汇总（§2.1.3-⑧）。

### 6.5 关键文件清单（本文档覆盖的源码）

| 文件 | 行数 | 角色 | 展开位置 |
|---|---|---:|---|---|
| `.vscode/launch.json` | 184 | 本文档入口 | §0/§1 |
| `scripts/tok_train.py` | 124 | BPE 训练 | §2.1 |
| `scripts/tok_eval.py` | 248 | 压缩率评估 | §2.2 |
| `scripts/base_train.py` | 908 | 基座预训练 | §2.3 |
| `scripts/base_eval.py` | 382 | 基座评估（CORE/BPB/sample） | §2.4 |
| `scripts/chat_sft_mini.py` | 146 | 离线 SFT | §2.5 |
| `scripts/chat_cli.py` | 113 | 对话 CLI | §2.6 |
| `nanochat/gpt.py` | 619 | 模型本体（形状核心） | §2.3.3-05/06/15 |
| `nanochat/optim.py` | 523 | MuonAdamW | §2.3.3-08 |
| `nanochat/tokenizer.py` | 440 | BPE 分词器 | §2.1.3-③⑤⑥⑦ / §2.5.3-⑥ |
| `nanochat/engine.py` | 393 | KV Cache 推理引擎 | §2.3.3-13 |
| `nanochat/common.py` | 353 | 公共设施 | §2.3.3-01 |
| `nanochat/core_eval.py` | 294 | CORE 指标 | §2.4.3-⑥ |
| `nanochat/checkpoint_manager.py` | 231 | 检查点 | §2.4.3-③ |
| `nanochat/dataset.py` | 179 | parquet 数据集 | §2.1.3-② |
| `nanochat/dataloader.py` | 150 | BOS best-fit 加载器 | §2.3.3-09 |
| `nanochat/flash_attention.py` | 163 | FA3/SDPA 统一接口 | §2.3.3-15 |
| `nanochat/loss_eval.py` | 65 | BPB | §2.3.3-12 |
| `nanochat/report.py` | 430 | 报告卡 | §2.1.3-⑧ |
| `tasks/common.py` / `tasks/customjson.py` | 124/65 | SFT 数据任务 | §2.5.3-③ |




