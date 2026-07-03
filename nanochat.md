# 分词script/tok_train.py 

> 以 `vocab_size=8192` 为例，走一遍脚本的关键变量和流程。

---

## 整体流程

```
Parquet 文件 → 流式读取 → 截断/限量 → Rust BPE 训练 → tiktoken.Encoding → 保存 → 缓存 token_bytes
```

---

## 1. 命令行参数 (args)

| 参数 | 示例值 | 含义 |
|---|---|---|
| `max_chars` | 2,000,000,000 (20亿) | 训练所用最大字符数，够数就停 |
| `doc_cap` | 10,000 | 单文档最大字符数，超过就截断 |
| `vocab_size` | **8192** | 最终词表总大小 |

---

## 2. SPECIAL_TOKENS（特殊标记，共 9 个，不参与 BPE 训练）

```python
SPECIAL_TOKENS = [
    "<|bos|>",            # 每个文档开头
    "<|user_start|>",     # 用户消息开始
    "<|user_end|>",       # 用户消息结束
    "<|assistant_start|>",# 助手消息开始
    "<|assistant_end|>",  # 助手消息结束
    "<|python_start|>",   # Python REPL 开始
    "<|python_end|>",     # Python REPL 结束
    "<|output_start|>",   # 输出开始
    "<|output_end|>",     # 输出结束
]
```

---

## 3. text_iterator() — 数据流

```python
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):   # 逐批读 parquet
        for doc in batch:                                 # 展平批次
            if len(doc) > 10000:                          # args.doc_cap
                doc = doc[:10000]                         # 截断
            nchars += len(doc)
            yield doc
            if nchars > 2_000_000_000:                    # args.max_chars
                return                                     # 够 20 亿字符就停
```

作用：**批次流 → 展平为文档流 → 截断 → 限总量 → 逐个 yield**，全程不把数据全加载入内存。

---

## 4. BPE 训练（Rust 实现）

```python
tokenizer = rustbpe.Tokenizer()
vocab_size_no_special = 8192 - 9 = 8183  # 扣掉特殊 token
tokenizer.train_from_iterator(text_iterator, 8183, pattern=SPLIT_PATTERN)
```

### 关键数字（vocab_size=8192）：

```
基础字节:    256 个（0x00~0xFF，覆盖所有 UTF-8 字节）
BPE 合并:    8183 - 256 = 7927 次
特殊 token:     9 个
─────────────────────────────
总词表:          8192 个
```

### SPLIT_PATTERN（GPT-4 风格正则，数字按 2 位分组而非 3 位）：

BPE 只在这条正则切出的"组"内部合并，不跨组。例如 `"Hello 123"` 切成 `"Hello" | " " | "12" | "3"`，BPE 不会把 `"o"` 和 `" "` 合并。

### text_iterator 的消费方式：

`train_from_iterator` 内部自己循环 `for text in text_iterator`，边拉数据边统计合并，不需要手动循环。

---

## 5. 组装 tiktoken.Encoding（推理用）

训练完成后，从 Rust tokenizer 提取三样东西：

```python
pattern = tokenizer.get_pattern()           # 切割正则
mergeable_ranks = tokenizer.get_mergeable_ranks()  # BPE 合并规则表
# mergeable_ranks 示例: {(b'h', b'e'): 256, (b'e', b'l'): 257, ...}
# 共 8183 条，每一条是"两段字节合并 → 新 token ID"
```

然后分配特殊 token ID，排在普通 token 后面：

```python
tokens_offset = 8183  # len(mergeable_ranks)

special_tokens = {
    "<|bos|>":            8183,   # 8183 + 0
    "<|user_start|>":     8184,   # 8183 + 1
    "<|user_end|>":       8185,   # 8183 + 2
    "<|assistant_start|>": 8186,  # 8183 + 3
    "<|assistant_end|>":  8187,   # 8183 + 4
    "<|python_start|>":   8188,   # 8183 + 5
    "<|python_end|>":     8189,   # 8183 + 6
    "<|output_start|>":   8190,   # 8183 + 7
    "<|output_end|>":     8191,   # 8183 + 8
}
```

最终组装：

```python
enc = tiktoken.Encoding(
    name="rustbpe",
    pat_str=pattern,          # 切割正则
    mergeable_ranks=...,      # BPE 合并规则 (8183条)
    special_tokens=...,       # 特殊 token 映射 (9条)
)
# 这个 enc 就能 encode/decode 了
```

### ID 布局总览：

```
  0 ~ 255   : 基础字节 token（BPE 起点）
256 ~ 8182  : BPE 合并出的子词 token（共 7927 个）
8183 ~ 8191 : 特殊控制 token（共 9 个）
```

---

## 6. token_bytes — BPB 查找表

```python
token_bytes = [len(decode([i]).encode("utf-8")) for i in range(8192)]
# token_bytes[i] = token_i 对应的 UTF-8 字节数
# 特殊 token 的 token_bytes = 0
```

### 作用：

计算 **BPB (bits per byte)** 时用——把"按 token 平均的 loss"转成"按字节平均的 loss"，消除词表大小对评估指标的影响，让不同词表的模型可以直接比较。

```
按 token  loss ≈ 按"个"算价格，大小不一不公平
按 byte   BPB  ≈ 按"斤"算价格，统一重量可比
```

存成 `token_bytes.pt`，推理时查表直接用，不用每次重新 decode。

---

## 7. 最终产物（保存到 tokenizer/ 目录）

| 文件 | 内容 |
|---|---|
| `tokenizer.json` | tiktoken 模型文件（合并规则 + 正则 + 特殊 token） |
| `token_bytes.pt` | `[8192]` int32 tensor，每个 token 的字节数 |

---

# 评估分词器 script/tok_eval.py

> 用三种分词器编码同一批文本，对比压缩率。**目的不是训练，是验证。**

---

## 整体流程

```
准备 7 种测试文本 → 用 gpt2/gpt4/ours 分别 encode → 计算 bytes/token → 对比表格 → 写入报告
```

---

## 1. 测试文本（7 种，覆盖不同场景）

| 文本 | 来源 | 测试目的 |
|---|---|---|
| `news_text` | 英文新闻（华盛顿邮报） | 英文通用文本 |
| `korean_text` | 韩文新闻 | 非英文/Unicode 文本 |
| `code_text` | Python 代码片段 | 代码压缩效果 |
| `math_text` | LaTeX 数学论文 | 科技/公式文本 |
| `science_text` | 光合作用科普 | 专业术语/长难句 |
| `train_text` | 训练集 parquet 第一批 | 训练数据上的表现（见过） |
| `val_text` | 验证集 parquet 第一批 | 验证数据上的表现（没见过） |

## 2. 数据获取方式

```python
# tok_eval.py  →  next() 只取第一批，几万字符
train_docs = next(parquets_iter_batched(split="train"))
train_text = "\n".join(train_docs)

# tok_train.py →  for 循环消费，20 亿字符
for batch in parquets_iter_batched(split="train"):
    ...
```

**同一个水管，用水量不同：** 评估只需要一小撮样本验证压缩率，训练才需要海量数据做 BPE 统计。

`next()` 取到的是**确定性**的（第一个 parquet 文件的第一个 row_group），不是随机。压缩率对取哪批文本不敏感，第一批就够。

---

## 3. 三种分词器

| 分词器 | 词表大小 | 来源 |
|---|---|---|
| `gpt2` | 50,257 | `RustBPETokenizer.from_pretrained("gpt2")` |
| `gpt4` | ~100K | `RustBPETokenizer.from_pretrained("cl100k_base")` |
| `ours` | 默认 32,768 | `get_tokenizer()` 加载刚训练的 |

gpt2/gpt4 只是**对比基准**（称重用的参考值），后续训练和推理用的始终是 `ours`。

---

## 4. 压缩率计算

```python
encoded = tokenizer.encode(text)           # 文字 → token id 序列
assert tokenizer.decode(encoded) == text   # 无损往返校验

ratio = len(text.encode('utf-8')) / len(encoded)
#      = UTF-8 字节数 / token 数
```

**ratio 越大越好**——说明每个 token 平均覆盖更多字节，同一段文本用更少 token 就能表示。

例如：`"Hello world"` = 11 字节，GPT-2 用 2 个 token → ratio = 11/2 = 5.5

---

## 5. 输出对比表格

每行对比一个文本类型，用颜色标注优劣：

```
Text Type   Bytes    GPT-2           Ours            Relative     Better
                     Tokens  Ratio   Tokens  Ratio   Diff %
------------------------------------------------------------------
news        1040     272     3.82    265     3.92     +2.6%       Ours
korean      680      195     3.49    210     3.24     -7.7%       GPT-2
code        585      150     3.90    148     3.95     +1.3%       Ours
...
```

`Relative Diff %` = `(基线tokens - 我们tokens) / 基线tokens × 100`，**正值 = 我们更好**（用更少 token 表示同一段文本）。

---

## 6. 与 tok_train.py 的关键区别

| | tok_train.py | tok_eval.py |
|---|---|---|
| 做什么 | **训练** BPE 分词器 | **评估** 分词器压缩率 |
| 数据量 | 20 亿字符（全部 8 个分片） | 一个 row_group（几 KB~几 MB） |
| 消费方式 | `for` 循环遍历 | `next()` 只取一批 |
| 耗时 | 几分钟（Rust BPE） | 几秒 |
| 输出 | 分词器模型文件 | 终端对比表格 + 报告 |
| 为什么这个量 | BPE 需要海量统计 | 压缩率在小样本上就很稳定 |

---

# 3. 预训练 base_train.py — 模型构建与注意力机制

> 以冒烟测试 2g (d4, head_dim=128, seq_len=512) 为例走一遍模型构建和数据流。

---

## 1. 模型维度推导

```python
depth=4, aspect_ratio=64, head_dim=128

base_dim  = 4 × 64 = 256          # 理想宽度
model_dim = 向上取整到128倍数 → 256  # 实际宽度（必须被 head_dim 整除）
num_heads = 256 / 128 = 2          # 注意力头数
```

| 变量 | 含义 |
|---|---|
| `depth` | Transformer 层数，**唯一核心旋钮**，其余自动推导 |
| `aspect_ratio` | 宽深比，默认 64（每加深一层，宽度 +64 维） |
| `base_dim` | 理想宽度 = depth × aspect_ratio |
| `model_dim` | 实际宽度，向上取整到 head_dim 的倍数 |
| `head_dim` | 每个注意力头的维度（粗细），默认 128 |
| `num_heads` | 注意力头数 = model_dim / head_dim |

---

## 2. 模型构建三步

```python
# Step 1: meta 设备构建（幽灵设备，只知形状不占内存）
with torch.device("meta"):
    model = GPT(config)

# Step 2: 在真实设备分配显存（数据是垃圾值，比 zeros 快）
model.to_empty(device=device)

# Step 3: 初始化权重（正态分布 N(0, 1/√fan_in)，截断到 ±3σ）
model.init_weights()
```

**meta 设备** = PyTorch 的"幽灵设备"：tensor 只有 shape/dtype，0 字节内存。先拿结构（参数量），再决定训练步数、batch 大小。

---

## 3. 模型结构（d4, model_dim=256 为例）

```
Input token ids [B, T] → Embedding [B, T, 256]
→ 4× Block（每个 = CausalSelfAttention + MLP）
    Attention: Q/K/V 投影 + 注意力计算 + 输出投影
    MLP:       Linear(256→1024) + ReLU² + Linear(1024→256)
→ RMSNorm → Linear(256→32768) LM Head → softcap tanh → loss
```

每层共 6 个权重矩阵，d4 模型 4 层 = 24 个，加 embedding/LM Head 约 3.4M 参数。

---

## 4. 多头注意力 (GQA)

### 标准 MHA vs GQA

```python
# 标准 MHA（base_train 当前用）：Q/K/V 头数相同
n_head = 4, n_kv_head = 4   # 一一对应

# GQA：K/V 头数比 Q 少，多个 Q 头共享同一组 K/V
n_head = 4, n_kv_head = 2   # 每 2 个 Q 头共享 1 组 KV
```

### 为什么 GQA

推理时 KV Cache 是显存瓶颈。KV 头数减半 = KV Cache 减半。LLaMA 70B 用 `n_head=64, n_kv_head=8`（8 个 Q 共享 1 组 KV）。小模型没必要，直接 MHA。

### Q/K/V 的关系

| 投影 | 角色 | 学的内容 |
|---|---|---|
| `c_q` (Query) | 查询 | "我当前这个位置**在找什么**" |
| `c_k` (Key) | 索引 | "我这个位置**有什么**可以给别人的" |
| `c_v` (Value) | 内容 | "如果有人关注我，我该**输出什么**" |

**三套独立权重，形状相同（[256,256]），但随机初始化不同 + 训练中梯度路径不同 → 自然分化出不同语义。** 初始化时随机，经过几万步梯度下降各自学到不同功能。

---

## 5. 注意力计算详解

### 数据流形状变化

```
输入:           [B, 512]           token ids
  ↓ Embedding 查表
嵌入:           [B, 512, 256]     B×T×C
  ↓ QKV 投影 (三套独立 Linear(256,256))
Q:              [B, 512, 2, 128]  B×T×H×D
K:              [B, 512, 2, 128]
V:              [B, 512, 2, 128]
  ↓ Q @ K^T (每个头独立算)
scores:         [B, 2, 512, 512]  B×H×T×T
  ↓ softmax (按行)
weights:        [B, 2, 512, 512]
  ↓ weights @ V
output:         [B, 2, 512, 128]
  ↓ 拼接两个头 + 输出投影
attn_out:       [B, 512, 256]
```

### Q @ K^T 详解

拿一个头看（忽略 batch）：`Q[512,128] @ K^T[128,512] → [512,512]`

```
attn_scores[i, j] = q_i · k_j   ← 位置 i 对位置 j 的"关注度分数"

每一行 = 当前位置对所有位置的关注度
值越大 = 越该关注
```

### 两次矩阵乘法 + 一次 softmax

```python
scores   = Q @ K^T      # 算相关性
weights  = softmax(scores)  # 转成概率权重（每行和=1）
output   = weights @ V  # 按权重取 V 的信息
```

---

## 6. 矩阵乘法 @ 运算规则

`@` = `torch.matmul`，看**最后两维**：

```python
# ① 1D @ 1D = 标量（点积）
[1,2,3] @ [4,5,6] = 1×4+2×5+3×6 = 32

# ② 2D @ 2D = 标准矩阵乘 (m,n)@(n,p)→(m,p)
[2,3] @ [3,4] → [2,4]

# ③ 1D @ 2D = 向量×矩阵 [n]@[n,p]→[p]
[256] @ [256,256] → [256]

# ④ 高维：前面维度当"批"，只算最后两维
[1,512,256] @ [1,512,256,128] 的写法不存在
而是 [1,512,256] 中每个 [256] 独立跟 W[256,256] 乘
```

---

## 7. view 拆分多头

```python
c_q = Linear(256, 256)    # 总输出 256 维，reshap 成两个头
q = c_q(x)                # [256]
q = q.view(2, 128)        # 前 128 个给头 0，后 128 个给头 1
```

`view` 是纯机械切段，不改变数据顺序。`256 = n_head × head_dim = 2×128`。

**主流模型都用硬切**（GPT-2/3/4、Gemini、DeepSeek 都一样），单个大 W 投影 + reshape 拆头。不同在头结构：ChatGPT 用标准 MHA，Gemini 用 GQA（KV 头更少），DeepSeek 用 MLA（KV 先压缩到潜空间）。但投影层永远是一个大 W，GPU 偏好大矩阵乘。

---

## 10. GPTConfig → GPT 模型构建（以冒烟测试 2g 为例）

### 参数推导

```python
# 冒烟测试输入:
depth=4, head_dim=128, aspect_ratio=64, max_seq_len=512
vocab_size=32768, window_pattern="SSSL"

# 推导:
base_dim  = 4 × 64 = 256
model_dim = 256 (向上取整到 128 倍数)
num_heads = 256 // 128 = 2
n_kv_head = 2  (等于 n_head，标准 MHA，不启用 GQA)

# GPTConfig:
config = GPTConfig(
    sequence_len=512,    # 最大上下文
    vocab_size=32768,    # 词表大小
    n_layer=4,           # 4 层 Transformer
    n_head=2,            # 2 个注意力头
    n_kv_head=2,         # KV 头=Q 头 (标准 MHA)
    n_embd=256,          # 嵌入维度
    window_pattern="SSSL" # 滑动窗口模式
)
```

### GPT(config) 内部创建了什么

执行 `GPT(config)` 时，`__init__` 在 meta 设备上运行（只有形状，不占内存）：

**① 滑动窗口：** `window_pattern="SSSL"` 按 4 层平铺，最后一层强制 L
```python
Layer 0: S (窗口=512//4=128)
Layer 1: S
Layer 2: S
Layer 3: L (512, 最后一层强制全上下文)
```

**② 词表填充：** 32768 对齐到 64 倍数 → 仍是 32768（刚好整除）

**③ 核心模块：**

| 模块 | 创建内容 | 形状 (d4) |
|---|---|---|
| `wte` | 词嵌入 | `Embedding(32768, 256)` |
| `h[0]~h[3]` | 4 个 Transformer Block | 每个 = Attention + MLP |
| `lm_head` | 输出投影 | `Linear(256, 32768)` |

**④ 每个 Block 内部：**

| 子模块 | 代码 | 形状 (d4) |
|---|---|---|
| `c_q` | Q 查询投影 | `Linear(256, 2×128) = Linear(256, 256)` |
| `c_k` | K 键投影 | `Linear(256, 2×128) = Linear(256, 256)` |
| `c_v` | V 值投影 | `Linear(256, 2×128) = Linear(256, 256)` |
| `c_proj` (attn) | 注意力输出投影 | `Linear(256, 256)` |
| `c_fc` (MLP) | MLP 升维 | `Linear(256, 1024)` (4×扩展) |
| `c_proj` (MLP) | MLP 降维 | `Linear(1024, 256)` |

**⑤ 额外结构：**

| 组件 | 说明 | 形状 |
|---|---|---|
| `resid_lambdas` | 每层残差缩放，init 1.15→1.05 线性衰减 | `[4]` |
| `x0_lambdas` | 每层混入初始嵌入权重，init 0.20→0.05 衰减 | `[4]` |
| `smear_gate` + `smear_lambda` | 前 token 嵌入泄漏（廉价 bigram） | `Linear(24,1)` + `[1]` |
| `backout_lambda` | 中层残差减去系数 | `[1]` |
| `value_embeds` | ResFormer VE，交替层启用（层 0,2 + 最后一层=层 3） | 3 个 `Embedding(32768, 256)` |
| `cos` / `sin` | 预计算 RoPE 频率 (10×seq_len=5120) | `[5120, 64]` 各一个 |

### 参数量估算

```
wte:              32768 × 256 =  8,388,608
lm_head:          32768 × 256 =  8,388,608
每层 Attention:
  c_q + c_k + c_v: 3 × 256×256 = 196,608
  c_proj:          256×256    =  65,536
  ─────────────────────────────────────
  每层 Attention 小计:          262,144
每层 MLP:
  c_fc:            256×1024   = 262,144
  c_proj:          1024×256   = 262,144
  ─────────────────────────────────────
  每层 MLP 小计:                524,288
每层 Block 小计: 262,144 + 524,288 = 786,432
4 层 Block:       4 × 786,432 = 3,145,728
value_embeds:     3 × 32768×256 = 25,165,824 (3 个 VE)
标量参数:          resid(4)+x0(4)+smear(1)+backout(1) ≈ 10
──────────────────────────────────────────
总计 ≈ 8.4M + 8.4M + 3.1M + 25.2M ≈ 45M 参数
```

> 注意：d4 模型虽小，但 VE（value embeddings）是大头——3 个 `[32768,256]` 占了 25M 参数，远远超过 4 层 Transformer 的 3.1M 矩阵参数。这是 nanochat 特有的"参数膨胀"设计。

---

## 8. 滑动窗口注意力

### 原理

标准因果注意力 O(T²)——序列翻倍，计算量翻 4 倍。滑动窗口：每个 token 只看前面 **固定窗口大小** 的位置。

```
L (Long/全上下文):   每个 token 看前面全部
S (Short/短窗口):    每个 token 只看前面 seq_len/4 个
```

### window_pattern

字符串按层循环平铺，最后一层强制 L：

```
window_pattern = "SSSL"  →  4 层模型：
  Layer 0: S (窗口 128)
  Layer 1: S (窗口 128)  
  Layer 2: S (窗口 128)
  Layer 3: L (窗口 512, 最后一层强制全上下文)
```

混合设计：大部分层用短窗口（局部信息已足够），间隔用全上下文（传递全局信息）。非 H100 GPU 建议 `--window-pattern=L`（SDPA 不支持窗口注意力）。

### 窗口在哪个维度

在**序列 T 维度**上，不是模型宽度。`attn_scores [512,512]` → 窗口限制让每行只看最近 128 列，前面的置 -inf（被 softmax 忽略）。

---

## 9. 所有参数都是训练出来的

d4 模型 ~3.4M 参数，全是随机初始化 → 几万步梯度下降训练出来的：

| 参数 | 形状 | 数量 |
|---|---|---|
| Embedding | [32768, 256] | 8.4M |
| 每层 Q/K/V 投影 ×3 | [256, 256] | ×4 层 |
| 每层 attn 输出投影 | [256, 256] | ×4 层 |
| 每层 MLP 升/降维 ×2 | [1024,256], [256,1024] | ×4 层 |
| LM Head | [32768, 256] | 8.4M |
| 标量参数 | ~10 个 | 忽略不计 |

Q 负责"提问"、K 负责"匹配"、V 负责"输出内容"——这不是人为规定，是数学结构（Q 和 K 通过点积耦合，V 通过加权耦合）加上梯度反向传播自然形成的分工。

---

## 11. CausalSelfAttention 详解

以 d4 (n_head=2, head_dim=128, n_embd=256) 为例。

### __init__ 初始化

```python
self.n_head   = 2     # Q 头数
self.n_kv_head = 2    # KV 头数 (标准 MHA)
self.head_dim = 128   # 每头 128 维 (256/2)
self.n_embd   = 256

# 四个 Linear 层（无 bias）
self.c_q    = Linear(256, 2×128) = Linear(256, 256)  # Q 投影
self.c_k    = Linear(256, 2×128) = Linear(256, 256)  # K 投影
self.c_v    = Linear(256, 2×128) = Linear(256, 256)  # V 投影
self.c_proj = Linear(256, 256)                        # 输出投影

# VE 门控 (仅 VE 层启用, d4 模型的层 0, 2, 3)
self.ve_gate = Linear(12, 2)  # 前 12 维 → 每头一个门控值
```

### forward 六步流程

**输入：** `x [B, T, 256]`, ve (可选), cos_sin, window_size, kv_cache

```python
B, T, C = x.size()  # 训练: B=1, T=512, C=256

# === 第 1 步：QKV 投影 ===
q = self.c_q(x).view(B, T, 2, 128)   # [1, 512, 2, 128]
k = self.c_k(x).view(B, T, 2, 128)   # [1, 512, 2, 128]
v = self.c_v(x).view(B, T, 2, 128)   # [1, 512, 2, 128]

# === 第 2 步：Value Embedding 注入（仅 VE 层）===
# ResFormer 风格：可学习的 value embedding 直接加到 V 上
if ve is not None:
    ve = ve.view(B, T, 2, 128)       # [1, 512, 2, 128]
    gate = 3 * sigmoid(ve_gate(x[前12维]))  # [1, 512, 2], 范围 (0, 3)
    v = v + gate * ve                # 门控加权混入

# === 第 3 步：RoPE 旋转位置编码 ===
# 对 Q 和 K 的最后 128 维 → 拆成 64 对 → 每对做 2D 旋转
# 效果：Q·K 内积自动包含相对位置信息，无需学习参数
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

# === 第 4 步：QK 归一化 ===
q, k = norm(q), norm(k)   # RMS 归一化（无学习参数）
q = q * 1.2               # 微小缩放增强注意力锐度
k = k * 1.2

# === 第 5 步：Flash Attention ===
# 训练路径（kv_cache=None）：
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
# 内部：Q@K^T → softmax → @V，自动处理因果 mask + 滑动窗口
#   头 0: [512,128]@[128,512]→[512,512] → softmax → @V[512,128] → [512,128]
#   头 1: 同上
#   结果: [1, 512, 2, 128]

# 推理路径（kv_cache 不为 None）：
# 增量追加 K/V 到缓存，只算新 token 的注意力

# === 第 6 步：拼回头 + 输出投影 ===
y = y.contiguous().view(B, T, 256)   # [1, 512, 2, 128] → [1, 512, 256]
y = self.c_proj(y)                    # [1, 512, 256] (零初始化，训练初期≈直通)
return y
```

### 归一化规则（该用哪些、不用哪些）

| Op | 归一化 | 原因 |
|---|---|---|
| 输入 `x` (Block 前) | RMSNorm ✓ | Pre-LN：先 norm 再进注意力 |
| Q / K | QK RMSNorm ✓ | 防止注意力 logit 爆炸，训练更稳定 |
| V | 不 norm | V 的尺度由权重自然控制 |
| 注意力输出 | 不 norm | 通过 `c_proj`（零初始化）后与残差加回 |
| MLP 输入 | RMSNorm ✓ | Block 内第二个 Pre-LN |

### 训练 vs 推理的关键区别

| | 训练 | 推理 |
|---|---|---|
| `kv_cache` | `None` | `KVCache` 对象 |
| K/V 来源 | 从 x 投影（完整 T） | 新 token 的 k/v 追加到缓存 |
| 注意力模式 | 一次性算 T×T | 增量：只算新 token 对历史的注意力 |
| 实现 | `flash_attn_func` | `flash_attn_with_kvcache` |
| 滑动窗口 | 通过 `window_size` 参数 | 同左 |