"""
使用自研 BPE Tokenizer 库训练分词器，风格类似 GPT-4 tokenizer。
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer  # Rust 实现的 BPE 分词器
from nanochat.common import get_base_dir  # 获取项目根目录
from nanochat.dataset import parquets_iter_batched  # 批量迭代 parquet 数据集

# -----------------------------------------------------------------------------
# 解析命令行参数
# Parse command line arguments

parser = argparse.ArgumentParser(description='训练 BPE 分词器 / Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='训练所用最大字符数 (默认: 20亿) / Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='单个文档最大字符数，超过则截断 (默认: 10,000) / Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='词表大小 (默认: 32768 = 2^15) / Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
# 打印训练配置参数
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# 文本迭代器：从数据集中流式读取文档，提供给分词器训练
# Text iterator

def text_iterator():
    """
    构建文本迭代器，流程如下：
    1) 将批次数据展平为单个文档迭代器
    2) 每个文档按 args.doc_cap 字符数截断
    3) 累计达到 args.max_chars 字符数后停止
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0  # 已处理的累计字符数
    for batch in parquets_iter_batched(split="train"):  # 仅使用训练集
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]  # 截断过长文档
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:  # 达到最大字符数限制后停止
                return
text_iter = text_iterator()  # 创建迭代器实例

# -----------------------------------------------------------------------------
# 训练分词器
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)  # 从文本迭代器训练 BPE
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# 将训练好的分词器保存到磁盘
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)  # 保存 tokenizer 配置和模型文件

# -----------------------------------------------------------------------------
# 快速内联正确性检查：编码后解码应完全一致
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)  # 编码为 token id 序列
decoded = tokenizer.decode(encoded)  # 解码回文本
assert decoded == test_text  # 确保编码-解码无损往返

# -----------------------------------------------------------------------------
# 额外步骤：缓存每个 token id 对应的 UTF-8 字节数映射 (token_id -> num_bytes)
# 用于高效计算 bits per byte (每字节比特数)。与常规的平均 loss 不同，
# bits per byte 指标不受词表大小影响，是评估分词器质量的主要指标之一。
# 验证集上的 bits per byte 是我们关注的核心指标。
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()  # 获取词表大小
special_set = set(tokenizer.get_special_tokens())  # 获取特殊 token 集合
# 预先解码所有 token，用于后续计算字节数
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # 该 token 的 Python 字符串表示 / the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # 特殊 token 不计入字节 / special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # 该 token 对应的 UTF-8 字节数 / number of bytes that make up this token
        token_bytes.append(id_bytes)
# 转换为 int32 tensor 并保存到磁盘
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# 输出训练报告日志
# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)  # 过滤掉特殊 token (字节数为0)
get_report().log(section="Tokenizer training", data=[
    vars(args), # 命令行参数 / argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {  # token 字节数统计信息 (仅非特殊 token)
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
