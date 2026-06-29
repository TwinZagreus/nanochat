"""
nanochat 的极简 FP8 训练 —— 仅支持张量级动态缩放。
Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

用约 150 行代码替代 torchao 的 Float8Linear（约 2000 行代码）。
我们只需要 "tensorwise" 配方（每个张量一个标量缩放因子），不需要 torchao
的全部功能（行级缩放、FSDP float8 all-gather、DTensor、张量子类分发表等）。
Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

FP8 训练的工作原理
How FP8 training works
======================
标准 Linear 层在前向中做一次矩阵乘法，在反向中做两次：
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 训练将这三个矩阵乘法各自包装为：
FP8 training wraps each of these three matmuls with:
  1. 计算：scale = FP8_MAX / max(|tensor|)，对每个操作数分别计算
     Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. 量化：fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
     Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. 通过 torch._scaled_mm 进行矩阵乘法（cuBLAS FP8 内核，比 bf16 快约 2 倍）
     Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. 反量化：_scaled_mm 内部使用逆缩放因子来处理
     Dequantize: _scaled_mm handles this internally using the inverse scales

关键洞察：torch._scaled_mm 和 float8 数据类型是 PyTorch 的内置功能。
torchao 只是对这些原语进行了编排。我们可以直接调用它们。
The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 数据类型选择
FP8 dtype choice
================
有两种 FP8 格式。我们按照标准惯例同时使用两者：
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4位指数，3位尾数，值域 [-448, 448]
    精度更高（更多尾数位），用于输入和权重。
  - float8_e5m2:   5位指数，2位尾数，值域 [-57344, 57344]
    范围更广（更多指数位），用于可能很大的梯度。

torch._scaled_mm 内存布局要求
torch._scaled_mm layout requirements
=====================================
cuBLAS FP8 内核要求特定的内存布局：
The cuBLAS FP8 kernel requires specific memory layouts:
  - 第一个参数 (A)：必须是行优先（连续内存）
    First argument (A):  must be row-major (contiguous)
  - 第二个参数 (B)：必须是列优先（B.t().contiguous().t()）
    Second argument (B): must be column-major (B.t().contiguous().t())
如果 B 是通过转置一个连续张量（如 weight.t()）获得的，它已经是列优先的
—— 无需复制。否则我们使用 _to_col_major()。
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

与 torchao 方法的区别
How this differs from torchao's approach
========================================
torchao 使用"张量子类"架构：Float8TrainingTensor 是 torch.Tensor 的子类，
它将 FP8 数据 + 缩放因子 + 元数据打包在一起。它实现了 __torch_dispatch__，
通过一个分发表拦截每个 aten 操作（mm、t、reshape、clone 等），
并以 FP8 感知的方式处理它们。当你调用
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
@ 运算符会分发到 aten.mm，后者被拦截并在背后路由到 torch._scaled_mm。
这需要约 2000 行代码，因为你需要为每个可能触及 FP8 张量的张量操作编写处理程序。
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

我们采用更简单的方法：一个单独的 autograd.Function（_Float8Matmul），它接收
全精度输入，内部量化为 FP8，调用 _scaled_mm，返回全精度输出。
标记了 @allow_in_graph，因此 torch.compile 将其视为一个不透明节点，而不是尝试追踪其内部。
We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

两种方法在 torch.compile 视角下的权衡：
The trade-off is in how torch.compile sees the two approaches:
  - torchao：compile 将张量子类分解（通过 __tensor_flatten__），
    将每个单独的操作（amax、scale、cast、_scaled_mm）视为独立的图节点。
    Inductor 可以将这些操作与周围的运算融合（例如将 amax 计算与前一层
    的激活函数融合）。
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

两者调用完全相同的 cuBLAS _scaled_mm 内核 —— GPU 矩阵乘法是完全相同的。
区别仅在于"胶水"操作（amax、scale、cast），这些操作相比矩阵乘法非常微小。
在实践中，这意味着我们的版本略微更快（更少的编译开销，没有张量子类分发成本），
但在 torch.compile 下可能产生细微不同的浮点舍入路径，因为 Inductor 生成了不同的图。
在 eager 模式下数值是逐位完全相同的。
Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# 避免在从全零张量计算缩放因子时出现除零错误
# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """使用张量级缩放将张量动态量化为 FP8。
    Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "张量级"（Tensorwise）意味着整个张量使用一个标量缩放因子（与"行级"（rowwise）
    相对，后者为每一行计算单独的缩放因子）。张量级更快，因为 cuBLAS 处理缩放；
    行级则需要 CUTLASS 内核。
    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    返回 (fp8_data, inverse_scale) 供 torch._scaled_mm 使用。
    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # 计算整个张量的最大绝对值
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # 缩放将 [0, amax] 映射到 [0, fp8_max]。使用 float64 进行除法以确保
    # torch.compile 和 eager 模式之间的数值一致性。
    # （torchao 也做同样的向上转型 —— 否则 compile/eager 可能会产生分歧）
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # 量化：缩放到 FP8 范围，饱和处理（clamp 防止类型转换时的溢出 ——
    # PyTorch 的默认行为是回绕，而非饱和），然后转换为 FP8
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm 期望的是缩放因子的*逆*（它在矩阵乘法期间通过乘以此值
    # 将 FP8 值转换回原始范围）
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """将二维张量的内存重新排列为列优先布局。
    Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm 要求其第二个操作数为列优先布局。
    技巧：转置 -> contiguous（强制以转置顺序进行复制）-> 再次转置。
    结果具有相同的逻辑形状，但步幅是列优先的，
    例如一个 [M, N] 张量得到步幅 (1, M) 而不是 (N, 1)。
    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph 告诉 torch.compile 将此操作视为不透明操作 ——
# dynamo 不会尝试将其分解为更小的操作。关于这与 torchao
# 张量子类方法的不同之处，请参阅模块文档字符串。
# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """为 Linear 层的三个 FP8 GEMM 操作定制的 autograd 函数。
    Custom autograd for the three FP8 GEMMs of a Linear layer.

    前向传播将输入和权重量化为 FP8，并保存量化后的张量和缩放因子供反向传播使用。
    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        """前向传播：将输入和权重以 FP8 格式相乘。

        将两个操作数量化为 e4m3（更高精度格式），调用 scaled_mm，
        返回全精度输出。
        """
        # 将两个操作数量化为 e4m3（更高精度格式）
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 是 [B, K] 连续的 = 行优先（适合第一个参数）
        # weight_fp8 是 [N, K] 连续的，所以 weight_fp8.t() 是 [K, N]，
        # 步幅为 (1, K) = 列优先（适合第二个参数，无需复制！）
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True 以较低精度累加点积。
            # 精度略低但速度明显更快。这是前向传播的标准做法；
            # 我们在反向传播中使用 False 以获得更精确的梯度。
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算输入梯度和权重梯度，均使用 FP8 matmul。

        分别计算 grad_input = grad_output @ weight 和
        grad_weight = grad_output.T @ input，各通过一次 FP8 scaled_mm 调用。
        梯度使用 e5m2（更广范围），权重使用 e4m3（更高精度）。
        """
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # 形状: [B, N] @ [N, K] -> [B, K]
        # 梯度使用 e5m2（更广范围），权重使用 e4m3（更高精度）
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 是 [B, N] 连续的 = 行优先，适合第一个参数
        # w_fp8 是 [N, K] 连续的 = 行优先，第二个参数需要列优先
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # 形状: [N, B] @ [B, K] -> [N, K]
        # go_fp8 是 [B, N] 连续的，我们需要 go.T = [N, B] 作为第一个参数。
        # 转置得到的是列优先，但第一个参数需要行优先，
        # 因此必须调用 .contiguous() 来物理重新排列内存。
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] 行优先 / row-major
        in_col = _to_col_major(in_fp8)    # [B, K] 列优先 / column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """可直接替换 nn.Linear 的 FP8 计算层。
    Drop-in nn.Linear replacement that does FP8 compute.

    权重和偏置保留其原始精度（例如 fp32/bf16）。
    只有矩阵乘法通过 _Float8Matmul autograd 函数以 FP8 执行。
    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    def forward(self, input):
        """前向传播：将输入转换为计算精度，展平为二维，通过 FP8 matmul 处理。"""
        # 将输入转换为 COMPUTE_DTYPE（通常是 bf16），因为 _scaled_mm 期望
        # 降低精度的输入，我们不再依赖 autocast 来做这件事。
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm 仅适用于二维张量，因此展平批次维度
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """从 nn.Linear 创建 Float8Linear，共享相同的权重和偏置。
        Create Float8Linear from nn.Linear, sharing the same weight and bias.

        使用 meta 设备避免分配临时权重张量 —— 我们在 meta 上创建模块外壳
        （仅有形状/数据类型，不分配内存），然后将 .weight 和 .bias 指向原始模块的参数。
        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """最小化配置，匹配 torchao 的 API。仅支持 tensorwise 配方。
    Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_name(recipe_name):
        """根据配方名称创建配置。当前仅支持 'tensorwise' 配方。"""
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """将模块中的 nn.Linear 层替换为 Float8Linear。
    Replace nn.Linear layers with Float8Linear throughout a module.

    以后序遍历模块树（子模块先于父模块），并将每个通过可选过滤器的
    nn.Linear 替换为 Float8Linear。新的 Float8Linear 共享原始权重和偏置张量
    —— 无复制，无额外内存占用。
    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: 要转换的根模块。 / Root module to convert.
        config: Float8LinearConfig（为 API 兼容性接受，仅支持 tensorwise）。
                Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: 可选过滤器 filter(module, fqn) -> bool。只有匹配的 Linear
                          层才会被转换。常见用法：跳过维度不能被 16 整除的层
                          （H100 上 FP8 矩阵乘法的硬件要求）。
                          Optional filter(module, fqn) -> bool. Only matching Linears
                          are converted. Common use: skip layers with dims not divisible by 16
                          (hardware requirement for FP8 matmuls on H100).
    """
    def _convert(mod, prefix=""):
        """递归辅助函数：遍历子模块并替换符合条件的 nn.Linear 层。"""
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
