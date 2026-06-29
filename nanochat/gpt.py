"""
nanochat 核心模块：GPT Transformer 语言模型（重写版，极简设计）
GPT model (rewrite, a lot simpler)

架构亮点:
  - RoPE 旋转位置编码（无学习参数，Q/K上施加旋转变换使内积自动包含相对位置）
  - QK 归一化（训练稳定性，比SDPA的softmax_temperature方案更优）
  - Untied Weights（词嵌入wte和输出投射lm_head不共享权重）
  - ReLU² 激活（ReLU→square，简单有效，等价于门控机制）
  - Post-embedding Norm（嵌入后立即RMS归一化）
  - RMSNorm 无可学习参数，Linear 无偏置（减少冗余参数）
  - GQA 支持（KV头数可少于Q头数，大幅节省推理KV缓存）
  - FA3 集成（Hopper+ GPU自动走FA3 kernel，其他GPU用PyTorch SDPA回退）

Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

关键设计: 权重存FP32（优化器精度），前向时Linear自动cast到COMPUTE_DTYPE做matmul
         嵌入(wte/ve)直接存COMPUTE_DTYPE省显存；fp16例外（GradScaler无法缩放fp16嵌入梯度）
============================================================================
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
# 统一Flash Attention接口：Hopper GPU → FA3 kernel，其他 → PyTorch SDPA自动回退
from nanochat.flash_attention import flash_attn

# ============================================================================
# GPTConfig: 模型配置（通过 --depth 一个旋钮推导所有超参数）
# model_dim = depth × aspect_ratio(64), n_head = model_dim / head_dim(128)
# ============================================================================
@dataclass
class GPTConfig:
    sequence_len: int = 2048  # 最大上下文长度
    vocab_size: int = 32768   # 词表大小 = 2^15，BPE分词器token数
    n_layer: int = 12         # Transformer层数（唯一手动指定的核心参数）
    n_head: int = 6           # number of query heads，Q注意力头数
    n_kv_head: int = 6        # number of key/value heads (GQA)，KV头数（<n_head时启用GQA）
    n_embd: int = 768         # 嵌入维度 = depth × aspect_ratio
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    # 滑动窗口模式: L=完整上下文(seq_len), S=短窗口(seq_len/4)，层间循环平铺，最后一层始终L
    window_pattern: str = "SSSL"


def norm(x):
    """RMS 归一化（无仿射变换参数）。
    仅做 x / sqrt(mean(x²) + epsilon)，比LayerNorm更轻量。
    RMS normalization (no learnable affine params). x / sqrt(mean(x²)), simpler than LayerNorm."""
    return F.rms_norm(x, (x.size(-1),)) # bf16下运行也OK / note that this will run in bf16, seems ok

# ============================================================================
# Linear: 自定义线性层——混合精度的核心
# 权重存FP32 → 前向时 cast 到输入激活的 dtype → 矩阵乘法在低精度下运行
# 效果 = autocast的混合精度，但完全显式可控
# ============================================================================
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    替代 torch.amp.autocast: 主权重FP32存储 → 优化器精度 → 前向自动转bf16做matmul"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """判断某层是否启用Value Embedding: 交替启用 + 最后一层必启用。
    Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    """RoPE旋转位置编码: 以最后维度d切半→对(d,d+1)维度对施加2D旋转→Q·K内积自动包含相对位置。
    RoPE: split last dim into d/2 pairs → rotate each pair as 2D vector → Q·K dot product encodes relative position."""
    assert x.ndim == 4  # 多头注意力 (B, T, H, D) / multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 对半切分最后维度 / split up last dim into two halves
    y1 = x1 * cos + x2 * sin # 2D旋转 / rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# ============================================================================
# CausalSelfAttention: 因果自注意力层 — Transformer的核心计算
# 流程: x→Linear投射Q/K/V → 可选VE门控注入 → RoPE → QK Norm → Flash Attention → 输出投射
# ============================================================================
class CausalSelfAttention(nn.Module):
    """因果自注意力层 — Transformer的核心计算单元。
    Causal Self-Attention — the core compute unit of the Transformer.

    流程/Pipeline: x → Linear投射Q/K/V → 可选VE门控注入 → RoPE位置编码 → QK Norm → Flash Attention → 输出投射
    """

    def __init__(self, config, layer_idx):
        """初始化因果自注意力层。
        参数/Params:
          config: GPTConfig模型配置
          layer_idx: 当前层索引(0-based)，用于VE和KV缓存管理"""
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head              # Q头数 / number of query heads
        self.n_kv_head = config.n_kv_head        # KV头数（GQA时< n_head） / number of KV heads (<n_head => GQA)
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head  # 每头维度 / head dim = C / n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0  # GQA约束: n_head是n_kv_head的整数倍
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)  # 输出投射（零初始化） / output projection (zero-init)
        # Value Embedding门控（ResFormer风格）: 取输入前12维 → Linear → sigmoid×3 → 范围(0,3)
        # Value Embedding gate (ResFormer-style): first 12 dims → Linear → sigmoid×3 → range (0,3)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """前向传播一个注意力层。
        参数/Args:
          x: (B, T, C) 输入激活 / input activations
          ve: (B, T, kv_dim) 或 None，Value Embedding / value embedding or None
          cos_sin: (cos, sin) RoPE旋转矩阵对 / rotary embedding pair
          window_size: (left, right) 滑动窗口大小，(-1,0)=全文 / sliding window, (-1,0)=full context
          kv_cache: KVCache对象，训练时None推理时传入 / KVCache object, None during training
        返回/Returns: (B, T, C) 注意力输出 / attention output"""
        B, T, C = x.size()

        # 投射输入→Q/K/V，形状(B,T,H,D)是FA3原生布局，无需转置！
        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value残差注入（ResFormer）: 用输入相关门控每头混合value embedding
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), 范围(0,3) / range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # RoPE施加旋转位置编码到Q和K / Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK归一化 / QK norm
        q = q * 1.2  # Q缩放，增强注意力锐度（Q/K各分担部分scale）/ sharper attention (split scale), TODO
        k = k * 1.2

        # Flash Attention：Hopper+用FA3 kernel，其他GPU自动SDPA回退
        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size: (left, right) = (N,0)→因果滑动窗口, (-1,0)→完整上下文
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # 训练路径：因果注意力 + 可选滑动窗口 / Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推理路径：用KV缓存增量计算注意力 / Inference: use flash_attn_with_kvcache
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 最后一层处理后推进缓存位置 / Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 重排多头输出→残差流 / Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP前馈网络：扩张→ReLU²激活→压缩。4x扩张 + ReLU.square()实现等效门控。
    MLP feed-forward: expand → ReLU² → project. 4x expansion + ReLU.square() = cheap gating."""

    def __init__(self, config):
        """初始化MLP。c_fc: 768→3072扩张，c_proj: 3072→768压缩（零初始化）。
        Init: c_fc expands 4x, c_proj projects back (zero-init)."""
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """前向传播: fc → ReLU → square → proj。
        Forward: fc → ReLU → square → proj. ReLU²近似GeLU的门控效果，更简单高效。"""
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer Block = CausalSelfAttention + MLP，各自Pre-LN(Norm→子层→残差加回)。
    Transformer Block = CausalSelfAttention + MLP, each with Pre-LN (norm → sublayer → residual add)."""

    def __init__(self, config, layer_idx):
        """初始化一个Transformer层。Init one transformer layer.
        参数/layer_idx: 当前层索引，用于VE配置和滑动窗口模式。"""
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """前向传播: Pre-LN → Attn → 残差加法 → Pre-LN → MLP → 残差加法。
        Forward: Pre-LN → Attn → residual add → Pre-LN → MLP → residual add."""
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        初始化GPT模型——注意: 此函数在meta device上下文中运行（参数仅为形状/类型占位，无实际数据）。
        所有实际数据初始化在init_weights()中完成。pad_vocab_size_to: 词表填充对齐因子(默认64)→利于DDP和Tensor Core。

        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # 计算每层滑动窗口大小 / Compute per-layer window sizes for sliding window attention
        # window_size: (left, right) = (-1,0)→完整上下文, (N,0)→滑动窗口
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # 词表填充→对齐到64倍数，利于DDP和Tensor Core / Pad vocab for efficiency (DDP, tensor cores)
        # This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # 每层可学习标量（受modded-nanogpt启发） / Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: 缩放每层残差流（init 1.0=无缩放）/ scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: 每层混入初始嵌入的权重（init 0.0=不混入）/ blends initial embedding back in at each layer (init 0.0 = disabled)
        # 分开参数以便优化器独立调度 / Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # 假初始化，真实在init_weights() / fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear机制: 将前一个token的嵌入泄漏到当前位置（廉价bigram信息）
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout机制: 最终归一化前减去缓存的中间层残差，去除低层特征（拼写/词法）→让lm_head看到更高层语义
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer风格): 交替层 + 最后一层必然启用 / Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # 为支持meta device初始化，这里预留RoPE缓存位置（假meta张量）。RoPE张量很小→预计算10倍序列长
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # rotary_seq_len = 10×序列长度（RoPE缓存小，过度计算10倍足够）/ 10X over-compute should be enough
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False=不存检查点 / not saved to checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        集中初始化全模型权重——所有参数在此统一初始化以保持清晰可追溯。
        Initialize the full model in this one function for maximum clarity.

        初始化策略/Init strategy:
          wte (embedding):     normal, std=0.8
          lm_head:             normal, std=0.001
          for each block:
            attn.c_q:          uniform, std=1/sqrt(n_embd)   # Q投影均匀分布避免离群值
            attn.c_k:          uniform, std=1/sqrt(n_embd)
            attn.c_v:          uniform, std=1/sqrt(n_embd)
            attn.c_proj:       zeros                          # 输出投影零初始化 (=无贡献开端)
            mlp.c_fc:          uniform, std=0.4/sqrt(n_embd)  # MLP×0.4降幅度
            mlp.c_proj:        zeros
          resid_lambdas: 1.15→1.05线性衰减(浅层更强) / linear decay from stronger early
          x0_lambdas: 0.20→0.05线性衰减(浅层更多x0混合) / more x0 blending early
          smear_gate: uniform(0, 0.02) 小正值→门控从接近中性开始
          backout_lambda: 0.2 常数
          ve_gate: uniform(0, 0.02) 小正值

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # 嵌入和反嵌入/Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer块: 均匀初始化，bound=√3×std 保证均匀分布与正态分布相同的标准差
        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # √3乘子确保均匀分布与正态分布同标准差 / sqrt(3) multiplier matches std of Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # 均匀分布避免离群值 / weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # 投影矩阵零初始化 / projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # MLP扩张×0.4降低初始幅度 / 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 每层标量参数初始化 / Per-layer scalars
        # resid线性衰减: 浅层更强残差(1.15)→深层逐步减弱(1.05)
        # Per-layer resid init: stronger residual at early layers (1.15), weaker at deep layers (1.05)
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # x0衰减: 浅层更多初始嵌入混合(0.20)→深层逐步减少(0.05)
        # Decaying x0 init: earlier layers get more input embedding blending (0.20→0.05)
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout标量和smear门控必须显式初始化 / must be explicitly initialized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings（类同c_v初始化: 均匀分布等标准差）
        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # VE门控权重: 小正值初始→门控从略高于中性开始
        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # RoPE旋转嵌入缓存 / Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 嵌入转COMPUTE_DTYPE省显存——优化器可容忍降低精度嵌入。
        # 例外: fp16需保持fp32嵌入，因为GradScaler无法缩放fp16梯度。
        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """预计算RoPE的cos/sin缓存。base=100K(近期的常见选择)→频率范围从1到1/base。
        Precompute RoPE cos/sin cache. base=100K → frequencies from 1 to 1/base.
        channel_range: [0, 2, 4, ..., head_dim-2] 隔步取通道
        inv_freq: 1/(base^(channel/head_dim)) 频率随通道指数衰减
        freqs: outer(时间步, 逆频率) → cos/sin → 加batch和head维度供广播"""
        # TODO: 是否提高base theta? 如100K是近期更常见的选择 / bump base theta more?
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # 按步取通道 / stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 时间步 / stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个(时间,通道)对的旋转频率 / calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # 加batch和head维度供广播 / add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        根据window_pattern计算每层的滑动窗口大小。
        Compute per-layer window sizes for sliding window attention.

        返回FA3 window_size参数的(left, right)元组列表:
          - left: 当前位置之前可关注的token数（-1=不受限）
          - right: 当前位置之后（0=因果，即只看到左边）
        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        模式串循环平铺到各层。最后一层始终=L（完整上下文）。
        Pattern string is tiled across layers. Final layer always gets L (full context).
        L=完整上下文(seq_len), S=短窗口(seq_len/4向上取整到128倍数，对齐FA3 tile).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # 字符→窗口大小映射 / Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # 向上取整到FA3 tile size / ceil to FA3 tile size (e.g. 2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # 模式串循环平铺 / Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 最后一层始终完整上下文 / Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        """获取模型所在设备(从wte.weight推断)。Get the device of the model (from wte.weight)."""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        估算每个token的FLOPs(前向+后向)。matmul: 前向2FLOPs→后向4FLOPs→共6FLOPs/参数。
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        注意力部分: 12*h*q*effective_seq_len (K@Q内积)，滑动窗口下effective_seq_len按层变化。
        Cleanest explanation: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        与Chinchilla公式~1%差异: 我们不计嵌入层查找FLOPs和softmax的exp/sum/div FLOPs。
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # 排除非matmul参数：嵌入和每层标量 / Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # 按层求和注意力FLOPs，考虑滑动窗口 / Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right)元组，取left / (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        返回分组参数数量，用于scaling law分析。不同论文约定不同:
        - Kaplan et al. 排除嵌入参数
        - Chinchilla 包含全部参数
        Return detailed parameter counts for scaling law analysis.

        返回按参数组分类计数的dict，方便下游实验哪种组合给出最干净的缩放定律。
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # 按组分别计数（对应setup_optimizers的分组）/ Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """构建优化器——按参数类型分组分配不同LR和优化器:
        - AdamW: 嵌入(lr=0.2)、lm_head(lr=0.004)、标量(lr=0.5)——所有LR按∝1/√dmodel缩放
        - Muon: Transformer矩阵参数(lr=0.02)——按shape分组堆叠
        - DDP时用DistMuonAdamW包装，单GPU用MuonAdamW
        Build optimizer with per-group LR and optimizer type:
        - AdamW: embeddings (lr=0.2), lm_head (lr=0.004), scalars (lr=0.5) — all scaled ∝1/√dmodel
        - Muon: transformer matrix params (lr=0.02) — grouped by shape for stacking
        - DistMuonAdamW for DDP, MuonAdamW for single GPU

        参数/Args:
          unembedding_lr: lm_head学习率 / learning rate for output projection
          embedding_lr: 嵌入学习率 / learning rate for token embeddings
          matrix_lr: Transformer矩阵学习率 / learning rate for transformer weights
          weight_decay: Muon的权重衰减 / weight decay for Muon
          scalar_lr: 标量参数学习率 / learning rate for scalar params
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # 按类别分离所有参数 / Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # AdamW参数学习率按∝1/√dmodel缩放（以768维模型为基准调优）
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # 构建参数组——所有字段显式指定 / Build param_groups with all required fields explicit
        param_groups = [
            # AdamW参数组（嵌入、lm_head、标量）
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # x0用更高beta1 / higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon参数组（Transformer矩阵参数，按shape分组堆叠）
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]   # 记录初始学习率供scheduler使用
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        GPT前向传播——训练和推理的统一入口。7步流程:
          ① RoPE缓存验证+动态扩展  ② 词嵌入+归一化+精度对齐
          ③ Smear(前一token嵌入泄漏) ④ 逐层Transformer处理(x0混合→resid缩放→VE注入→Block)
          ⑤ Backout(减去中层残差)   ⑥ 最终归一化→lm_head→logit softcap
          ⑦ loss(训练) 或 logits(推理)
        """
        B, T = idx.size()

        # ① RoPE: 获取旋转嵌入缓存，序列超长时动态扩展(e.g. 评估长prompt)
        # Dynamically expand the rotary cache if the sequence is longer than expected (e.g. eval prompts)
        if T > self.cos.size(1):
            head_dim = self.config.n_embd // self.config.n_head
            self.rotary_seq_len = T + 128  # 略多留点padding / add a bit of padding
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # 有KV缓存时需偏移到当前缓存位置 / if kv cache exists, offset to current position in cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()  # KV缓存推理时需偏移到当前缓存位置
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # 截取当前序列长度 / truncate cache to current sequence length

        # ② 嵌入: 查表 → 转精度 → 归一化 / Embed the tokens: lookup → cast → norm
        x = self.transformer.wte(idx) # 嵌入当前token / embed current token
        x = x.to(COMPUTE_DTYPE) # 确保激活值在计算精度(bf16时一般为no-op) / ensure activations in compute dtype
        x = norm(x)

        # ③ Smear机制: 将前一个token嵌入泄漏到当前位置（廉价bigram信息）
        # gate = λ·σ(W·x_t[:24]) ∈ (0,λ)，将位置t-1的嵌入按门控混入位置t
        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # 训练/朴素生成: 序列完整→快速切片 / Training: full sequence, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)  # x_t += gate_t × x_{t-1}
        else:
            # KV缓存推理: 从缓存读取前嵌入，存储当前供下一步 / KV cache inference: read prev, store current
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill阶段: 对位置1+应用smear，同训练模式 / Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode阶段: 单token，用缓存的之前嵌入 / Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # ④ Transformer主体: 逐层通过N个Block，注入VE、滑动窗口
        # Forward the trunk: N layers with VE injection, sliding windows
        x0 = x  # 保存初始归一化嵌入供逐层x0混合 / save initial normed embedding for per-layer x0 blending
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # 在中点层缓存残差 / cache residual at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            # resid_lambda缩放残差 + x0_lambda回混初始嵌入
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Value Embedding注入（仅交替层 + 最后一层，ResFormer风格）
            # Value Embedding injection (alternating layers + last layer, ResFormer-style)
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x  # 缓存中点残差供backout使用
        # ⑤ Backout: 减去中层残差→移除拼写/词法级底层特征→让lm_head仅看到高层语义
        # Subtract mid-layer residual to remove low-level features before logit projection
        # => removes spelling/lexical surface-level features → lm_head sees higher-level semantics
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # ⑥ lm_head计算logits + softcap平滑截断到[-15,15]
        # Forward the lm_head + softcap squashes logits to [-15,15] for stability
        softcap = 15 # 平滑截断logits到[-softcap, softcap] / smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) ← 巨大张量，大量显存 / very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # 切掉填充部分 / slice to remove padding
        logits = logits.float() # 转FP32→softcap和交叉熵计算数值稳定 / switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # 平滑截断/squash / squash the logits

        # ⑦ 训练：返回loss / 推理：返回logits
        # Training: compute loss / Inference: return logits
        if targets is not None:
            # 训练: 给定targets→计算交叉熵损失 / training: compute cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理: 直接返回logits / inference: return logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        朴素自回归流式生成——用于参考/调试/正确性验证（Engine.generate()更高效）。
        Naive autoregressive streaming inference.

        流程: 每步→forward→取最后位置logits→top_k截断→温度缩放→采样→yield→拼接→循环
        Process: forward → last-position logits → top_k filtering → temperature → sample → yield

        特点: batch=1, 无KV Cache(O(N²)复杂度), tokens和yield的token都是Python int列表。
        To make it super simple, let's assume:
        - batch size is 1, no KV Cache (O(N²) complexity)
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # 加batch维度 / add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size) — 只取最后位置的logits / only take last position
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 找出第k大的作为阈值 / find k-th largest as threshold
                logits[logits < v[:, [-1]]] = -float('Inf')  # 低于阈值的→概率=0 / set below threshold to -inf
            if temperature > 0:
                logits = logits / temperature  # 温度缩放 / temperature scaling
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 贪婪解码/温度=0→取最大 / greedy decoding
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
