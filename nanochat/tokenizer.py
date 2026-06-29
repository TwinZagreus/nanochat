"""
GPT-4 风格的 BPE 分词器 (Byte-Pair Encoding Tokenizer)。

提供两种实现:
1) HuggingFace Tokenizer: 可同时训练和推理，但代码非常混乱
2) RustBPE Tokenizer: 使用 RustBPE 训练 + tiktoken 高效推理（项目默认选择）

BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

# 特殊标记列表：BOS分隔文档 + 对话角色标记 + 工具调用标记
# List of special tokens: BOS delimits documents + conversation role tokens + tool call tokens
SPECIAL_TOKENS = [
    # BOS标记：每个文档的开头，用于分隔文档 / every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # 以下标记仅在微调(SFT)时使用，将对话渲染为token序列 / tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # 用户消息开始 / user messages
    "<|user_end|>",   # 用户消息结束
    "<|assistant_start|>", # 助手消息开始 / assistant messages
    "<|assistant_end|>",   # 助手消息结束
    "<|python_start|>", # 助手调用Python REPL工具 / assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # Python REPL输出返回给助手 / python REPL outputs back to assistant
    "<|output_end|>",
]

# 注意: 此分割模式与GPT-4不同——用\p{N}{1,2}替代\p{N}{1,3}
# 因为不想让小词表(32K)浪费太多token在数字上。验证了2是最优的数字分组大小。
# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# 通用GPT-4风格分词器 — 基于HuggingFace Tokenizer实现（训练+推理一体）
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """基于HuggingFace Tokenizer的轻量封装，提供便捷工具方法。
    Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        """用HuggingFace底层tokenizer对象初始化。
        Initialize with a HuggingFace underlying tokenizer object."""
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        """从HuggingFace预训练分词器加载(如"gpt2")。
        init from a HuggingFace pretrained tokenizer (e.g. "gpt2")"""
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """从本地目录加载分词器(如"out/tokenizer")。
        init from a local directory on disk (e.g. "out/tokenizer")"""
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """从文本迭代器训练BPE分词器。配置GPT-4风格的分割+字节级预分词。
        train from an iterator of text. Configures GPT-4-style splitting + ByteLevel pre-tokenization."""
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # 字节回退：未知字符→UTF-8字节，确保无UNK / needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None  # 归一化器：不归一化
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4风格正则分割
        # Pre-tokenizer: GPT-4 style
        # GPT-4使用的正则模式，将文本在BPE训练前分割成组
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # 注意: 模式从\p{N}{1,3}改为\p{N}{1,2}，因为怀疑1,3对小模型和小词表有害（浪费token空间）
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # （但尚未验证！TODO）
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # HF要求用Regex()包装正则！ / huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel（与ByteLevel预分词器配对使用）
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None  # 后处理器：无
        tokenizer.post_processor = None
        # Trainer: BPE训练器
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # 无最小频率要求 / no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # 启动训练 / Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        """获取词表大小。Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        """获取所有特殊标记列表。Get list of all special tokens."""
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        """Token ID → token字符串。Decode a single token ID to string."""
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        """编码单个字符串为token序列。可选前置/后置特殊标记。
        encode a single string.
        prepend/append can be either a string of a special token or a token id directly.
        num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)"""
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        """精确匹配编码单个特殊标记→token ID。
        encode a single special token via exact match"""
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        """获取BOS token ID。HF模型BOS各不同：1)找<|bos|> 2)找<|endoftext|>(GPT-2) 3)找不到则报错。
        Different HuggingFace models use different BOS tokens and there is little consistency
        1) attempt to find a <|bos|> token  2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        3) if these fail, it's better to crash than to silently return None"""
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        """编码文本→token序列。支持单字符串或字符串列表。
        Encode text to token sequence. Supports single string or list of strings."""
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        """直接调用对象→等价于调用encode()。Call object directly → same as encode()."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """Token ID序列→字符串解码（保留特殊标记）。
        Decode token IDs to string (keep special tokens)."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        """保存分词器到磁盘(JSON格式)。
        save the tokenizer to disk"""
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# RustBPE + tiktoken 组合分词器：用RustBPE高速训练，用tiktoken高效推理（项目默认使用）
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """基于tiktoken做高效推理、rustbpe做BPE训练的轻量封装。
    Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        """用tiktoken Encoding对象和BOS token初始化。
        Initialize with tiktoken Encoding object and BOS token."""
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """从文本迭代器训练BPE分词器：1)RustBPE训练 2)构建tiktoken编码(用于高效推理)。
        train from iterator: 1) train using rustbpe, 2) construct tiktoken encoding for efficient inference."""
        # 1) RustBPE训练（特殊标记不参与训练，稍后在__init__中插入）
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) 构建tiktoken Encoding：合并规则→(bytes→rank)映射 + 特殊标记映射
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int]（token字节 → 合并优先级排名）
            special_tokens=special_tokens, # dict[str, int]（特殊标记名称 → token ID）
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """从本地目录加载(pickle格式tiktoken编码对象)。
        Load from local directory (pickle-serialized tiktoken encoding)."""
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """从tiktoken预训练名称加载(如"gpt2")。tiktoken称文档分隔符为<|endoftext|>，
        但nanochat统一称<|bos|>(beginning of sequence)。两者功能相同——都表示新序列开始。
        Load from tiktoken pretrained name. In tiktoken the document delimiter is called <|endoftext|>,
        which is confusing because it's almost always PREPENDED, not appended. NanoChat uses <|bos|> for clarity."""
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """获取词表大小。Get vocabulary size."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """获取特殊标记集合。Get set of special tokens."""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """Token ID → token字符串。Decode a single token ID to string."""
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """编码特殊标记→token ID（带LRU缓存，最多缓存32个）。
        Encode special token to token ID (with LRU cache, max 32 entries)."""
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """获取BOS token ID。Get BOS token ID."""
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """编码文本→token序列。支持单字符串(tiktoken普通编码)或字符串列表(批量多线程编码)。
        prepend/append可以是特殊标记字符串或直接给token ID。num_threads仅对批量编码生效。
        Encode text to token ids. text can be either a string or a list of strings.
        prepend/append: special token string or int token ID. num_threads used only for batch encoding."""
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # 性能可优化？目前可接受 / TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        """直接调用对象→等价于encode()。Call object directly → same as encode()."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """Token ID序列→字符串解码。Decode token IDs to string."""
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        """保存tiktoken编码对象到磁盘(pickle格式)。
        save the encoding object to disk"""
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        渲染对话为token序列+训练mask——SFT最关键的预处理步骤。
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).

        mask含义:
          - mask=1: Assistant输出部分 → 参与交叉熵损失计算
          - mask=0: 用户消息/系统提示/特殊标记/工具输出 → 损失计算时忽略
        mask=1: Assistant output (participates in loss), mask=0: User/system prompt/special tokens/tool output (ignored)

        Returns:
        - ids: list[int] 渲染后的token ID序列 / list of token ids of this rendered conversation
        - mask: list[int] 等长mask数组，Assistant需训练的位置为1 / mask of same length, 1 for tokens the Assistant is expected to train on.

        格式/Fomat: [BOS|<user_start>|文本|<user_end>|<assistant_start>|回复|<assistant_end>]×N
                     mask= 0   0           0       0         0               1       1
        """
        # ids, masks列表 + 辅助函数构建它们 / ids, masks and a helper function to build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            """添加token_ids到ids列表，同时扩展等长的mask值。Append tokens + extend mask."""
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 如果有system消息→合并到第一条user消息中
        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # 需要对话结构调整，深拷贝避免修改原始数据 / some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # 获取所有需要的特殊标记ID / fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # 开始逐消息编码 / now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 角色交替检查: user→assistant→user→... 防止数据错误
            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content可以是简单字符串，也可以是parts列表(含工具调用)
            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 纯文本回复 → 全部参与损失 / simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # 文本部分 → 参与损失 / string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # Python工具调用 → 参与损失（模型需学会何时调用工具）
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # Python输出 → 不参与损失（推理时来自真实Python执行器，非模型生成）
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # 截断到max_tokens（防止超长对话OOM）/ truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """调试辅助函数: 用终端颜色可视化render_conversation的分词结果。
        Small helper function useful in debugging: visualize the tokenization of render_conversation.
        绿色=mask=1(参与损失)，红色=mask=0(忽略)，灰色=token ID。
        Green=mask=1 (train), Red=mask=0 (ignore), Gray=token ID."""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        强化学习(RL)阶段使用: 渲染对话但移除最后一条Assistant消息，用前缀引导模型生成新回复。
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        不同于SFT，RL不需要返回mask（因为我们只关心模型生成的新token）。
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # 移除最后一条Assistant消息 / pop the last message (of the Assistant)
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # 避免修改原始数据 / avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # 原地移除最后一条Assistant消息 / remove the last message (of the Assistant) inplace

        # 编码剩余对话 / Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # 添加<|assistant_start|>引导模型开始生成 / prime the Assistant for a completion
        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat项目级别的便捷函数：获取分词器和token字节张量
# nanochat-specific convenience functions

def get_tokenizer():
    """获取项目默认分词器(RustBPETokenizer)，从base_dir/tokenizer目录加载。
    Get the default project tokenizer (RustBPETokenizer) from base_dir/tokenizer directory."""
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    """加载token字节张量(token_bytes.pt)，由tok_train.py生成，用于tokenizer行为的额外分析。
    Load token bytes tensor (token_bytes.pt), written by tok_train.py, for additional token analysis."""
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
