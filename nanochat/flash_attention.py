"""
Unified Flash Attention interface with automatic FA3/SDPA switching.
统一Flash Attention接口: Hopper(H100 SM90)+bf16 → FA3 kernel，否则 → PyTorch SDPA回退

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.

GTX1630(SM75) → SDPA回退(无FA3)，功能相同但无滑动窗口加速

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)  # Training
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)    # Inference
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """尝试加载 Flash Attention 3 (需要 Hopper GPU, SM90)。
    Try to load Flash Attention 3 (requires Hopper GPU, sm90).
    非Hopper(SM90)架构 (包括Ada SM89, Blackwell SM100) 或CUDA不可用时返回 None, 走SDPA回退。"""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # FA3 kernel仅针对Hopper(SM90)编译
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        # Ada(SM89), Blackwell(SM100)等架构需SDPA回退, 直到FA3重新编译
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """决定是否使用FA3: 根据可用性、手动覆盖、计算dtype综合判断。
    Decide once whether to use FA3, based on availability, override, and dtype.
    优先顺序: 手动覆盖 > FA3可用性 (Hopper + SM90 + bf16) > SDPA回退"""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        # FA3 Hopper kernel仅支持bf16/fp8; fp16/fp32须用SDPA回退
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    带滑动窗口支持的 SDPA attention实现。q, k, v 格式为 (B, H, T, D)。
    SDPA attention with sliding window support. q, k, v are (B, H, T, D).

    三种路径:
      1) 完整上下文 (无滑动窗口限制, Tq==Tk) → is_causal=True 直接调用
      2) 单token生成 (Tq==1) → is_causal=False, 滑动窗口裁剪kv
      3) 分块推理/滑动窗口训练 → 显式构建bool注意力掩码
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length / 完整上下文, 等长序列: 直接用因果SDPA
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation / 单token生成: 非因果, 裁剪滑动窗口
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total / 窗口=左侧需包含的key数
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference / 需显式构建掩码: 滑动窗口或分块推理
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    # 分块推理时(Tq≠Tk), is_causal对齐不上cache位置 → 构建显式bool掩码
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left) / 滑动窗口限制 (左侧)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D) / SDPA格式转换
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D) / 转回原始格式


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache / SDPA回退: 手动管理KV缓存
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch / 假设batch内位置一致

    # Insert new k, v into cache (in-place, matching FA3 behavior) / 写入新k,v到缓存 (原地, 匹配FA3行为)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens / 获取当前位置+新token的完整缓存
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D) / 转置为SDPA格式
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D) / 转回原始格式


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
