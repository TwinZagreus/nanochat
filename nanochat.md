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