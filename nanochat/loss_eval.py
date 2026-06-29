"""
A number of functions that help with evaluating a base model.
评估工具: BPB (bits per byte) 计算，词表大小无关的损失度量。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    BPB 评估: bits per byte = 总损失(nats) / (ln2 × 总字节数)。

    为什么不直接用 mean loss:
      普通 loss 依赖于词表大小——换一个词表loss值就变了，无法公平对比。
      BPB 将损失按目标token的字节数归一化: 每个token贡献 loss × bytes(token)
      → 消除词表大小影响 → 不同词表的模型可直接对比BPB。

    特殊处理:
      1) 特殊token(<|bos|>等)字节数=0 → 不参与计算
      2) ignore_index=-1的target → 跳过(不索引token_bytes)
      3) 分布式时all_reduce汇总各rank的nats和bytes

    Args:
      token_bytes: (vocab_size,) int32, 每个token id的UTF-8字节数, 0=不计入
    """
    # record the losses / 累积损失
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T) / 逐token损失
        loss2d = loss2d.view(-1) # flatten / 展平
        y = y.view(-1) # flatten / 展平
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32 / MPS不支持int64的<0比较, 需转int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # 部分target为ignore_index(-1)时的处理: 跳过这些位置
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            # 任何<0的target token都应忽略, 不可用于索引token_bytes
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            # 有效target映射到对应字节数; 忽略的target贡献0字节
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            # 快速路径: 无忽略target, 直接索引token_bytes
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks / 跨所有rank求和汇总
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return / 移至CPU, 计算BPB并返回
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
