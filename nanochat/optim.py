"""
高效混合优化器: AdamW + Muon 组合。
通常嵌入层和标量参数使用AdamW, 矩阵参数使用Muon。
提供两个版本: MuonAdamW(单GPU) 和 DistMuonAdamW(分布式)。

A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

改自: https://github.com/KellerJordan/modded-nanogpt
来自@karpathy和@chrisjmccormick的进一步贡献。
Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
条件torch.compile: 当TORCH_COMPILE_DISABLE=1时变为无操作(no-op)。
允许脚本在导入此模块之前禁用编译(例如在缺少MSVC编译器的Windows上)。

Conditional torch.compile: becomes a no-op when TORCH_COMPILE_DISABLE=1.
This allows scripts to disable compilation before this module is imported
(e.g., on Windows without MSVC compiler).
"""
import os as _os

def _conditional_compile(fn=None, *, dynamic=False, fullgraph=True):
    """
    条件性编译装饰器: 通过环境变量TORCH_COMPILE_DISABLE控制是否启用torch.compile。
    可用作 @_conditional_compile 或 @_conditional_compile(dynamic=True)。

    Conditional compile decorator that checks TORCH_COMPILE_DISABLE env var.
    Can be used as @_conditional_compile or @_conditional_compile(dynamic=True).
    """
    def decorator(fn):
        if _os.environ.get("TORCH_COMPILE_DISABLE"):
            return fn
        return torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph)
    if fn is None:
        return decorator
    return decorator(fn)

# -----------------------------------------------------------------------------
"""
经典AdamW优化器, 融合内核版本。
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@_conditional_compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    算法公式: p *= (1-lr×wd) → m = β₁·m+(1-β₁)·∇L → v = β₂·v+(1-β₂)·∇L²
             → 偏差校正 m̂=m/(1-β₁ᵗ), v̂=v/(1-β₂ᵗ) → p -= η×m̂/(√v̂+ε)
    0-D CPU tensor: 值变形状不变 → torch.compile 复用已编译的图
    """
    # Weight decay (decoupled, applied before the update)，解耦式: 直接衰减参数而非通过梯度
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)     # m = β₁·m + (1-β₁)·grad
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)  # v = β₂·v + (1-β₂)·grad²
    # Bias corrections: 修正初始阶段矩估计偏低问题
    bias1 = 1 - beta1_t ** step_t  # →1 as t→∞
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t  # √v̂+ε
    step_size = lr_t / bias1                       # η/(1-β₁ᵗ)
    p.add_(exp_avg / denom, alpha=-step_size)      # p -= η × m̂/(√v̂+ε)

# -----------------------------------------------------------------------------
"""
Muon优化器, 改编并简化自modded-nanogpt。
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

背景: Newton-Schulz迭代用于计算G的零次幂/正交化。选择五次迭代,其系数
被优化以最大化零点处的斜率。为最小化步数,即使迭代不再在整个区间上完全
收敛到1,继续增加零点处斜率在经验上也是有效的。因此此迭代不产生UV^T,
而是产生类似US'V^T的结果,其中S'是对角矩阵,S_{ii}' ~ Uniform(0.5, 1.5),
这与SVD(G=USV^T)的UV^T相比对模型性能完全无害。

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Polar Express符号方法: Newton-Schulz迭代的替代方案,具有更好的收敛性质,用于正交化。
Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon方差衰减: 逐神经元/逐列自适应学习率,归一化正交化后的更新尺度
(Muon的输出在不同神经元间尺度不均匀)。
NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

nanochat实现中的一些变化 / Some of the changes in nanochat implementation:
- 使用更简单、更通用的参数分组和堆叠方法
  Uses a simpler, more general approach to parameter grouping and stacking
- 使用单一融合内核: 动量 -> polar_express -> 方差衰减 -> 更新
  Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- 不对模型架构做任何假设(例如注意力权重不一定融合成QKVO格式)
  Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Polar Express系数(针对num_iters=5, safety_factor=2e-2, cushion=2计算)
# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# 来源: https://arxiv.org/pdf/2505.16932 / From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@_conditional_compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    四步流程: ① Nesterov动量 → ② Polar Express正交化(梯度矩阵→近似正交矩阵)
            ③ NorMuon方差衰减(逐神经元归一化) → ④ 谨慎权重衰减(同号才衰减)
    参考: Polar Express https://arxiv.org/pdf/2505.16932, NorMuon https://arxiv.org/pdf/2510.05491
    """

    # ① Nesterov动量: m=μ·m+(1-μ)·∇L → g=∇L+μ·m(外推梯度)
    # ① Nesterov momentum: m=μ·m+(1-μ)·∇L → g=∇L+μ·m
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # ② Polar Express正交化
    # Polar express
    # 可用时转为bf16加速; 否则跳过(fp16因指数范围受限,此处不稳定)
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # 高矩阵(Tall matrix): 行数>列数,用X.mT@X
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # 宽矩阵(Wide matrix): 列数>=行数,用X@X.mT(原始公式)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # ③ NorMuon方差衰减 / Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # ④ 谨慎权重衰减 + 参数更新: 仅当梯度方向与参数同号时衰减(避免干扰有用方向)
    # Cautious weight decay + parameter update: only decay params whose sign matches gradient
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0  # 同号才衰减 / same-sign mask
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.
    Muon + AdamW 混合优化器(单GPU版): 2D矩阵→Muon(动量+正交化), 嵌入/标量/1D→AdamW(自适应)

    AdamW - Fused AdamW optimizer step. https://arxiv.org/abs/1711.05101
    Muon - MomentUm Orthogonalized by Newton-schulz. https://kellerjordan.github.io/posts/muon/
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings: Muon不能用于嵌入/输出层/0-1D参数; 4D卷积核压平后三维即可
    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        对组内每个参数单独执行AdamW更新: 惰性初始化状态 → 填充0-D张量 → 调用融合内核。
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # 状态惰性初始化 / State lazy init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # 用当前超参值填充0-D CPU张量 / Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        对组内所有参数统一执行Muon更新(堆叠后批量处理以提高效率)。
        惰性初始化状态 → 填充0-D张量 → 调用融合内核。
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # 获取或创建组级缓冲区(为了方便存储在第一个参数的状态中)
        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # 每个单独参数的一阶动量 / Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # 二阶动量缓冲区分因式存储,按行或按列 / Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # 堆叠梯度和参数(注意: 假设所有参数形状相同) / Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # 用当前超参值填充所有0-D CPU张量 / Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # 单一融合内核: 动量 → Polar Express正交化 → 方差衰减 → 参数更新 / Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # 将更新拷贝回原始参数 / Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    分布式混合优化器: 2D矩阵参数用Muon, 其他用AdamW。
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    单优化器算法细节见MuonAdamW。此类添加分布式通信以支持多GPU训练(无需PyTorch DDP)。
    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    设计目标 / Design Goals:
    - 通信与计算重叠(异步操作) / Overlap communication with computation (async ops)
    - 跨rank分片优化器状态以最小化内存(ZeRO-2风格) / Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - 尽可能将小张量批量合并为单次通信操作 / Batch small tensors into single comm ops where possible

    通信模式(3阶段异步) / Communication Pattern (3-phase async):
    使用3阶段结构最大化通信与计算的重叠:

        Phase 1: 启动所有异步reduce操作 / Launch all async reduce ops
            - 发起所有reduce_scatter/all_reduce操作 / Kick off all reduce_scatter/all_reduce operations
            - 不等待,让它们在后台运行 / Don't wait - let them run in background while we continue

        Phase 2: 等待reduce, 计算更新, 启动gather / Wait for reduces, compute updates, launch gathers
            - 对每个组: 等待其reduce完成 → 计算更新 → 启动gather
            - 按顺序处理组,早期gather可与后续计算重叠

        Phase 3: 等待gather, 拷贝回原参数 / Wait for gathers, copy back
            - 等待所有gather完成 / Wait for all gathers to complete
            - 将更新后的参数拷贝回原始张量(仅Muon) / Copy updated params back to original tensors (Muon only)

    AdamW通信(ZeRO-2风格) / AdamW Communication (ZeRO-2 style):
    - 小参数(<1024个元素): all_reduce梯度,每个rank更新完整参数。
      优化器状态虽复制但参数极小(标量/偏置)。
    - 大参数: reduce_scatter梯度,每个rank获得1/N梯度,仅更新该切片,
      然后all_gather更新后的切片。优化器状态(exp_avg, exp_avg_sq)分片存储,
      每个rank仅存储其切片的优化器状态。要求param.shape[0]能被world_size整除。

    Muon通信(堆叠+分块) / Muon Communication (stacked + chunked):
    - Muon组内所有参数必须具有相同形状(调用者责任)。
    - 将K个参数堆叠为单个(K, *shape)张量以便高效通信。
    - 将K个参数分配给N个rank: 每个rank"拥有" ceil(K/N)个参数。
    - reduce_scatter堆叠的梯度,每个rank获得其分块。
    - 每个rank仅对其拥有的参数计算Muon更新。
    - all_gather将更新后的参数返回给所有rank。
    - 优化器状态(momentum_buffer, second_momentum_buffer)按块分片。
    - 填充: 如果K不能整除,零填充到 (ceil(K/N) * N) 用于通信,拷贝时忽略填充。

    缓冲区复用 / Buffer Reuse:
    - 对于Muon,分配stacked_grads作为reduce_scatter输入,然后复用同一缓冲区
      作为all_gather的输出(stacked_params)。因不需要同时持有两个缓冲区,这样节省内存。

    参数 / Arguments:
        param_groups: 字典列表, 每个包含:
            - 'params': 参数列表
            - 'kind': 'adamw' 或 'muon'
            - AdamW组还包含: 'lr', 'betas', 'eps', 'weight_decay'
            - Muon组还包含: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """启动AdamW组的异步reduce操作。返回包含逐参数信息的字典。 / Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # 小参数: all_reduce(无需scatter/gather) / Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # 大参数: reduce_scatter,被world_size整除 / Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """启动Muon组的异步reduce操作。返回信息字典。 / Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # 堆叠梯度并零填充至padded_num_params / Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter获取本rank的分块 / Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """等待reduce完成 → 计算AdamW更新 → 为大参数启动gather。 / Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # 小参数操作完整参数; 大参数仅操作本rank切片 / For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # 状态惰性初始化 / State lazy init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # 填充0-D张量并运行融合内核 / Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # 大参数需要all_gather / Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """等待reduce完成 → 计算Muon更新 → 启动gather。 / Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # 本rank拥有多少个参数? / How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # 获取或创建组级状态 / Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # 构建all_gather的输出缓冲区 / Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # 填充0-D张量并运行融合内核 / Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:  # 填充槽位清零 / Zero out padding slots
            updated_params[num_owned:].zero_()

        # 复用stacked_grads缓冲区作为all_gather的输出 / Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """等待所有gather完成并将Muon参数拷贝回原张量。 / Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: 从堆叠缓冲区拷贝回各个参数 / Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        """
        分布式更新步骤(3阶段异步流水线):
        ① 启动所有异步reduce → ② 等待reduce并计算更新 + 启动gather → ③ 等待gather并拷贝回.
        Distributed step with 3-phase async pipeline:
        ① Launch all async reduces → ② Wait reduces + compute updates + launch gathers → ③ Wait gathers + copy back.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: 启动所有异步reduce操作 / Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")  # 未知优化器类型 / Unknown optimizer kind

        # Phase 2: 等待reduce, 计算更新, 启动gather / Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")  # 未知优化器类型 / Unknown optimizer kind

        # Phase 3: 等待gather, 拷贝回原参数 / Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
