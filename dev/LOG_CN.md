# 实验日志

持续记录实验和发现的运行总结。始于 2026 年 1 月 7 日左右。

---

## 2026-05-05：d12 预训练的 DyT 实验（负面结果）

尝试用 [DyT](https://arxiv.org/abs/2503.10622) 替换归一化层，在 d12 规模的预训练上测试。受 X 上[热度](https://x.com/LodestoneRock/status/2050367217087512953)影响。

- DyT 使用 `gamma * tanh(alpha * x) + beta`，有可学习的标量 `alpha` 和逐通道的 `gamma`/`beta`
- 为注意力层与其他归一化位置分别添加了 alpha 初始化器，遵循论文宽度相关的启发式方法（除非被覆盖）
- 添加了可选的 embedding DyT，以及论文中 LLM 专属的 `sqrt(d_model)` embedding 缩放

尝试的每一种变体，包括大量参数调优，均未能超过 master 上的 d12 baseline，即使以步数为 x 轴也一样。此外，吞吐量（tokens per second）下降了约 10%。

---

## 2026-03-24：Parameter-Golf 想法扫描（负面结果）

审查了 `openai/parameter-golf`，寻找可移植到 nanochat 预训练的小型/简单想法（不增加代码臃肿）。缓存笔记在 `knowledge/parameter_golf.md`。

### 理由

parameter-golf 排行榜是以下内容的有用来源：
- 微小的架构调整
- 短时的优化器/调度技巧
- Muon 相关的系统想法

但该仓库的大部分优化目标是完全不同的：
- 适配 16MB 产物
- 在 8×H100 上训练不超过 10 分钟
- 以压缩/bpb 作为评估指标

因此只有少数想法看起来值得在 nanochat 上尝试。

### 尝试的想法

**1. LeakyReLU(0.5)^2**
- 将 MLP 中的 `relu^2` 替换为 `leaky_relu(x, 0.5)^2`
- **结果：** 每步质量略好，但略慢。wall clock 上更差。

**2. Partial RoPE**
- 仅对每个头维度的前四分之一应用 rotary embeddings
- **结果：** 略差。

**3. LN Scale**
- 在 attention 和 MLP 之前将每个 block 的归一化输入乘以 `1/sqrt(layer_idx+1)`
- **结果：** 没有帮助。

**4. Orthogonal init**
- 将非零 transformer 矩阵切换到正交初始化，同时保持零初始化的输出投影
- **结果：** 没有帮助。

**5. XSA (Exclusive Self Attention)**
- 仅在非 VE 的最深 3 层上实现 XSA，使其投影到纯 `v` 路径而非 `v + VE`
- **结果：** 步质量略好，但 wall clock 上不占优。不值得在热注意力路径中增加额外计算。

### 备注

- EMA/SWA 之前已尝试过（我忘了记录），没有帮助。
- 双元哈希嵌入在更早的时候已探索，确实有一定帮助，但在较大规模下增加的参数/VRAM/复杂度不合理。参见上方的 1 月 27-28 日条目。

### 结论

这次扫描没有找到任何廉价的可移植 parameter-golf 技巧，能在我们关心的指标（wall clock 到能力的时间）上明显改善 nanochat。

---

## 2026-03-04：移除 autocast，显式 dtype 管理，fp16 GradScaler

用单一的 `COMPUTE_DTYPE` 全局变量替代了代码中所有的 `torch.amp.autocast`，实现了显式的 dtype 管理。还增加 fp16 训练支持和 GradScaler。

### 动机

autocast 是"我们无法控制的魔法"——它通过内部的 allowlist 决定哪些 op 在哪种精度下运行。对这个代码库来说，autocast 做的很少：唯一真正转换的是 `nn.Linear` 的权重（fp32 到 bf16）用于 matmul。`F.rms_norm`、`F.cross_entropy` 和 Flash Attention 都已经自行处理 dtype。通过显式管理精度，我们获得了精细粒度的控制（比如可以试验 fp32 norm），并消除了一层不必要的抽象。

### 变更内容

**核心机制** (`nanochat/common.py`, `nanochat/gpt.py`)：
- `COMPUTE_DTYPE` 根据硬件自动检测：SM 80+ → bf16，pre-Ampere → fp32，CPU/MPS → fp32。可通过 `NANOCHAT_DTYPE` 环境变量覆盖。
- 自定义 `Linear(nn.Linear)` 类在前向时将权重转换为输入 dtype：`F.linear(x, self.weight.to(dtype=x.dtype))`。这是替代 autocast 的唯一机制。
- Embedding 在初始化时转换为 `COMPUTE_DTYPE`（节省内存）。例外：fp16 时 embedding 保持 fp32，因为 GradScaler 无法 unscaling fp16 梯度。
- 在 `GPT.forward()` 中显式将 embedding 输出转换到 `COMPUTE_DTYPE`（bf16 路径无操作，fp16 路径生效）。
- RoPE cos/sin 缓存使用 `COMPUTE_DTYPE` 而非硬编码的 bf16。

**Autocast 移除**（11 个文件）：
- 删除了 `--dtype` CLI 标志、`ptdtype` 变量、`autocast_ctx` 定义，以及所有 `with autocast_ctx:` 块。涉及文件：`base_train.py`、`chat_sft.py`、`chat_rl.py`、`chat_cli.py`、`chat_eval.py`、`chat_web.py`、`base_eval.py`、`engine.py`、`bench_train_toks.py`、`test_e2e_pipeline.py`。

**fp16 + GradScaler** (`base_train.py`, `chat_sft.py`)：
- `scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None`
- 反向传播：`scaler.scale(loss).backward()` 对比普通的 `loss.backward()`
- 累积后：`scaler.unscale_(optimizer)` → 分布式 inf 同步（通过 `scaler._found_inf_per_device(optimizer)` all-reduce 用 `ReduceOp.MAX`）→ `scaler.step(optimizer)` → `scaler.update()`
- bf16/fp32 路径零开销（scaler 为 None，kernel 内无分支）。

**FP8 修复** (`nanochat/fp8.py`, `base_train.py`)：
- `Float8Linear.forward` 显式将输入转换为 `COMPUTE_DTYPE`（之前依赖 autocast）。
- `disable_fp8` 上下文管理器现在在评估期间切换 Float8Linear 时创建自定义 `Linear`（而非原始 `nn.Linear`）。

**Flash Attention** (`flash_attention.py`)：
- FA3 Hopper kernel 不支持 fp16 或 fp32，所以 `USE_FA3`（模块级常量，import 时一次性解析）返回 False，回退到 SDPA。

---

## 2026-03-04：数据集升级：FineWeb-EDU 100B → ClimbMix 400B

将预训练数据集从 FineWeb-EDU 100B 切换到 ClimbMix 400B。这是对 nanochat GPT-2 速通时间最大的一次单次改进，将其从 **2 小时 46 分钟降到 2 小时 1 分钟**——减少了 27%。

### ClimbMix 是什么？

ClimbMix 400B 是一个精心筛选的 400B-token 预训练混合数据集，托管在 HuggingFace 的 `karpathy/climbmix-400b-shuffle`。来自 [NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)。它是高质量网页文本、代码、数学和其他来源的混合，设计目标是比单独使用 FineWeb-EDU 更好地作为通用预训练数据集。

### 变更内容

- **数据集**：`karpathy/fineweb-edu-100b-shuffle` → `karpathy/climbmix-400b-shuffle`（可用分片从 1823 个增至 6543 个，支持未来更长训练）
- **数据目录**：`base_data/` → `base_data_climbmix/`（与旧数据清晰分离）
- **模型深度**：d26 → d24。ClimbMix 训练效率更高，更小的模型即可达到 GPT-2 水平
- **分片数量**：现在只需要约 150 个数据分片（约 7B tokens）即可达到 GPT-2 能力
- **评估 tokens**：从 40 批加倍到 80 批，以获得更稳定的验证 loss 估算
- **旧版回退**：在 `list_parquet_files()` 中添加了迁移警告，检测旧的 `base_data/` 目录并优雅回退，使现有用户在 `git pull` 后看到清晰的升级说明

### 背景

这是第六次尝试在 CORE 得分上超越 FineWeb-EDU——前五次全部失败（见下方 2 月 17 日、2 月 10 日、1 月 12 日的条目）。ClimbMix 是第一个令人信服地超越它的数据集，且差距足够大，可以同时将模型从 d26 缩小到 d24。

---

## 2026-03-02：SoftCap 调优

快速实验在 d24 规模上调优 logit softcap。尝试了 5 到 30。5 非常糟糕，其余基本相当，除了 20 表现最好。小幅但稳健的改进：验证 loss 改善了约 1e-3（0.716 → 0.715）。设为默认值。

## 2026-02-19：混合专家模型 MoE（负面结果）

实现了 DeepSeekV3 风格混合专家层，用作密集 MLP 的替代品。MoE 分支可以工作并改善了每步验证 loss，但由于 MoE 的开销，在 wall clock 上不是净胜（至少在我们关注的约 GPT-2 能力的规模上）。

### 实现

遵循 DeepSeekV3 并以 torchtitan 为参考：
- **8 个路由专家，top-2 路由**，使用 sigmoid 门控（非 softmax）
- **1 个共享专家**（密集 MLP 处理所有 token，遵循 DeepSeekV3）
- **无辅助损失的负载均衡**（DeepSeekV3 的 expert bias nudging）
- **等 FLOP 尺寸**：`expert_hidden_dim = round(4 * dim / (top_k + num_shared) / 128) * 128`，使每 token 活跃 FLOPs 与密集 MLP 匹配
- **`torch._grouped_mm`** 用于在单个 kernel 中将 token 分发到专家（而非 Python for 循环）
- **3D 专家权重张量** `(num_experts, hidden, dim)`——Muon 的 Polar Express 在最后两维上操作，每个专家独立正交化
- **活跃参数计数**用于缩放定律（仅 `top_k + shared` 专家，非全部 8 个）

### 容易实现的部分
- 核心 MoE 前向传播：router、按专家排序 tokens、grouped matmul、scatter 回原位。概念上干净。
- 共享专家：只是 `nn.Linear` MLP，所有 token 上都运行，与路由路径并排。
- 3D 专家参数 + Muon：只需要修复 `second_momentum_buffer` 形状以保留前导维度。
- 负载均衡：DeepSeekV3 的 bias nudging 简单有效（约 10 行代码）。

### 困难的部分
- **`torch._grouped_mm` 怪癖**：要求 bf16（非 fp32）、右操作数按列存储、int32 累积偏移。API 无文档，只能靠试错发现。
- **Token 数量填充**：torchtitan 为更好的 grouped_mm 吞吐将每个专家的 token 数填充到对齐倍数（bf16 时为 8）。我们用纯 PyTorch 方法和复制 torchtitan 的 Triton kernel 两种方式实现了这一点。两者都干净编译（0 graph breaks），但在约 65K tokens、8 个专家的情况下（每个专家已获得约 8K tokens 是很好对齐的），填充开销（gather/scatter）实际上反而将 MFU 从 35% 倒退到 33%。已回退。
- **FP8 + MoE**：`torch._grouped_mm` 不支持 FP8。有一个单独的 `torch._scaled_grouped_mm` API，需要按行缩放（非按张量像我们的 `Float8Linear`）。权重梯度的反向传播需要按组的按列缩放，torchao 用自定义 Triton kernel 实现。我们深入研究了（见 `dev/moe_fp8.md`）但没有实现——要么依赖 `torchao.prototype`（不稳定），要么写约 200 行自定义 autograd + 量化代码。部分 FP8 支持存在：共享专家的 `nn.Linear` 层确实被转换了，但路由专家（3D `nn.Parameter`）保持 bf16。

### 结果
- d18：MFU 从约 46% 降至约 35%（grouped_mm 调度 + token 排序开销很大）
- 每步验证 loss 的改善无法补偿吞吐损失
- 综合 wall clock 上为净负

### 未来可探索的方向
- **路由专家的 FP8**：使用 `torch._scaled_grouped_mm` 配合自定义 `_Float8GroupedMatmul` autograd 函数，权重梯度回退到 bf16（避免按组按列的 Triton kernel）。

真正需要的是一个融合的 "FlashMoE" kernel，一次性处理 routing + expert dispatch + matmul（类似 FlashAttention 为注意力做的那样），包含所有必要特性。目前还不存在这种 kernel。用现有的 PyTorch 原语裸写 MoE 非常痛苦——大量排序、gather、scatter 和布局转换围绕实际计算。

### 结论
MoE 目前在 nanochat 不值得。代码臃肿很大（moe.py、router、shared expert、负载均衡、optimizer 修复、FP8 缺口、活跃参数计数），且在我们关注的规模上 wall clock 性能更差。根本问题是 grouped_mm 调度开销吞掉了稀疏性带来的 FLOP 节省，至少在我们模型规模和序列长度下如此。

---

## 2026-02-17：预训练数据：FineWeb（负面结果）

用 vanilla FineWeb 代替 FineWeb-edu 数据集。显著惊人地更差：
- d26（GPT-2）：CORE 0.2602 → 0.2241

这是第五次在 CORE 得分上未能超越纯 FineWeb-EDU 的尝试。

---

## 2026-02-17：预训练数据混合实验（负面结果）

尝试了 [hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT](https://huggingface.co/datasets/hynky/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT)，一个 FinePDFs、DCLM 和 FineWeb-EDU 的混合数据集。在两个规模的模型上都略差：
- d26（GPT-2）：CORE 0.2602 → 0.2549
- d18：CORE 0.199 → 0.192

这是第四次在 CORE 得分上未能超越纯 FineWeb-EDU 的尝试。

---

## 2026-02-16：SFT 脚本升级

将 `chat_sft.py` 更新到与 `base_train.py` 功能对齐，并根据 SFT 扫描调整了设置。

### 调优
- **优化器热启动**（`--load-optimizer=1`，默认开启）：通过 `checkpoint_manager.py` 中的新 `load_optimizer_state()` 加载预训练动量缓冲区。加载后 LR 重置为新的 SFT 值。加载优化器效果略好但不显著。
- **LR 调度**：用 warmup/constant/warmdown（匹配 `base_train.py`）替换了"constant 80%, linear to 0"。与预训练类似，warmdown ratio 0.5 效果最好。`--init-lr-frac` 从 1.0 微调到 0.8。
- **LR 调优**：尝试调优所有单独的 LR（例如 SFT 是否偏好更低的 embedding LR 等），所有尝试均得到负面结果。
- **数据混合**：MMLU epochs 1→3，GSM8K epochs 2→4（扫描确认最佳）。epoch 数现在可通过 `--mmlu-epochs` / `--gsm8k-epochs` 配置。不过未来可能移除。

### 质量改进与 bug 修复
- **超参数继承**：SFT 现在默认从预训练检查点元数据继承 batch sizes 和 LR（CLI 覆盖仍然有效）。同时将 `total_batch_size` 保存到 `base_train.py` 检查点元数据中。
- **GC 管理**：在 step 1 后禁用 Python GC，避免约 500ms 的暂停（每 5000 步手动回收一次），与基础预训练相同。
- **ChatCORE 评估**：SFT 期间定期评估（`--chatcore-every=200`），覆盖全部 6 个任务，记录到 wandb。
- **MFU**：使用 `get_peak_flops()` 获取实际 GPU 峰值而非硬编码的 H100 值。
- 移除了 `--dry-run` 和 `--dtype` 标志。所有 rank 现在都参与检查点保存。

---

## 2026-02-05：自动批大小缩放

### 背景

之前 `--total-batch-size` 硬编码为 `2**19 = 524,288` ≈ 0.5M tokens。这是 d12 的最优设置，但在为 d26（GPT-2 级别）重新调优时，发现最优值更接近 `2**20 = 1,048,576` ≈ 1M tokens。这是可以预期的——更大的模型偏好更大的最优 batch size。然而我们必须确保所有 `--depth` 设置都能以某种有原则的方式获得自己的最优 batch size。这里参考了 Cerebras 的 "Power Lines" 论文（[arXiv:2505.13738](https://arxiv.org/abs/2505.13738)），其中发现 **Bopt ∝ D^0.383**（其中 D 是训练 token 数而非参数数量！）。思路是在 d12 上调优出最优 batch size，然后用这个幂律外推到更大的模型。0.383 的指数意味着 batch size 增长缓慢：10× token 仅需要约 2.4× 的 batch size。对于 nanochat 的计算最优训练（D ∝ N 通过 `--target-param-data-ratio`），这意味着更深的模型自然需要更大的 batch size。

### 实现

添加了 `--total-batch-size=-1`（现为默认值）来自动计算最优 batch size，参考点 d=12 模型 B=2^19（经验验证），随模型深度自动调整。

### 结果

| Depth | Scaling Params | Target Tokens | Auto Batch |
|-------|---------------|---------------|------------|
| d=8   | 42M           | 0.44B         | 2^18 = 262K |
| d=10-16 | 70M-235M    | 0.7B-2.5B     | 2^19 = 524K |
| d=18-26 | 324M-918M   | 3.4B-9.6B     | 2^20 = 1.05M |
| d=32-50 | 1.7B-6.2B   | 17.6B-65.6B   | 2^21 = 2.1M |

特别地，这匹配了经验观察：d26 偏好约 2^20 而 d12 偏好约 2^19。

### 额外实验：batch size ramp

尝试了 batch size 渐进增长。最简单的实现通过将每个微批次切分来"欺骗"训练循环，在训练早期更频繁调用 optimizer.step()（前 x% 训练过程中从 1/8 → 1/4 → 1/2 → full batch，配合 sqrt LR 缩放）。还需要 torch.compile 预热阶段来预编译所有切片大小以避免训练中的重编译尖峰。虽然想法合理且观察到了小幅度收益，但不足以证明引入的代码复杂度。未合并。

---

## 2026-02-05：SwiGLU 激活（负面结果）

用 SwiGLU 替换了 ReLU² MLP 激活。SwiGLU 使用三个投影而非两个，为匹配参数量需将隐藏维度从 4× 缩放至 8/3×。在 d12 和 d24（GPT-2 规模）上做了测试。所有指标（步效率、wall clock、FLOPs）均更差。ReLU² 仍然是 nanochat 最优选择。**未采纳。**

---

## 2026-02-03：翻转 Muon MLP LR 乘数（PR #492）

测试了翻转 Muon 中基于形状的 LR 启发式方法。原代码将高矩阵（输入投影如 `c_fc`）乘以约 2× LR。翻转版改为将宽矩阵（输出投影如 `c_proj`）乘以约 2× LR，这符合经典的 fan-in/fan-out 缩放惯例。d12 快速实验：**略差。未采纳。**

---

## 2026-02-03：跳过 AdamW 隔步更新

受 modded-nanogpt 启发，尝试仅奇数迭代步进 AdamW，Muon 每步都步进。思路是小参数不需要像大权重矩阵那样频繁更新，跳过可同时节省计算和通信。d12 上 tok/sec 快约 2%，但每步 loss 略差。以 wall clock 为 x 轴衡量时，综合略差。**未采纳。**

---

## 2026-02-02：FP8 训练 with torchao

集成 FP8 训练，在 H100 GPU 上使用 `torchao.float8` 加速 Linear 层 matmul。H100 的 FP8 tensor core 理论上有约 2× matmul 吞吐量。代价是量化开销（计算 scale 和转换张量到/从 FP8）。

### 背景

torchtitan 报告部分实验有 25-28% 的加速。之前 2026 年 1 月的尝试只对 `lm_head` 做 FP8（仿 modded-nanogpt），结果仅 1% 加速且 +2GB 内存（torch.compile 交互脆弱）。本次尝试使用 torchao 对**所有** Linear 层进行 FP8 转换。

### 结果

**微基准（d26 MLP, 65536×1664 @ 1664×6656）：**
| 方法 | Forward | Fwd+Bwd | 加速 |
|------|---------|---------|------|
| BF16 + compile | 2.00ms | 4.79ms | 1.00x |
| FP8 rowwise + compile | 1.84ms | 4.55ms | 1.08x |
| FP8 tensorwise + compile | 1.45ms | 4.06ms | **1.38x** |
| FP8 rowwise (no compile) | 2.89ms | 21.86ms | 0.23x ❌ |

torch.compile 是**必须的**。没有它 FP8 因融合量化 ops 反而慢 4×。

**完整训练（d26）：**
| 配置 | tok/sec | vs baseline |
|------|---------|-------------|
| BF16 baseline | 630K | 1.00x |
| FP8 rowwise | 564K | 0.90x ❌ |
| FP8 tensorwise | 740K | **1.17x** ✓ |

显存也减少约 9GB（激活值以 FP8 而非 BF16 存储）。

但每个 FP8 步骤精度略低，匹配能力需训练更长时间。在 d24 规模扫描发现能力匹配的增速更接近 **5%**。我们的 LLM 在约 d24 规模可能还是太小，无法充分享受 FP8 对更大模型的好处。

### 关键经验
1. **Tensorwise >> Rowwise**——Rowwise 计算每行 scale，开销超过收益
2. **过滤小层**——维度不能被 16 整除的层必须跳过（FP8 硬件要求）
3. **更大模型受益更多**——d12 用 FP8 更慢；d26+ 开始见效
4. **能力匹配的实际加速更低**——每步精度略低

---

## 2026-01-29：Hyperball/MuonH 实验（负面结果）

探索了来自博客文章的 Hyperball 优化。将权重约束在半径为 R 的球面。做了多组实验：MuonH 替代矩阵参数、LR 扫描、可学习 RMSNorm scale、AdamH 替代 lm_head 等。所有变体均无法超越 baseline。文章对 AdamH 在 `lm_head` 上的具体实现细节不详。经过几小时的调优和调试，这不是 nanochat 的开箱即用胜局。可能以后会再次尝试。**未采纳。**

---

## 2026-01-28：回退双元哈希嵌入

从代码库中移除了 bigram embeddings（engram-lite）。在更大规模（d25）上改善微乎其微，以 wall clock 衡量时完全消失。它还膨胀了 VRAM 使用。额外的参数和复杂度不合理。

---

## 2026-01-27：双元哈希嵌入（Engram-lite）

受 [DeepSeek Engram 论文](https://arxiv.org/abs/2601.07372) 和 modded-nanogpt PR #201 启发，探索了 N-gram 记忆模块。

### 背景
Engram 论文引入"条件记忆"作为 MoE 的补充——用 O(1) 哈希查找检索静态 N-gram 模式，替代通过计算重构它们。核心洞察：transformers 在早期层浪费计算去"模拟对模式（如命名实体、公式化短语）的检索"，而这本可以是简单的查表。

### 尝试的内容
1. **完整 Engram 模块（论文设计）**：上下文感知的门控 + 哈希双元。嵌入作为 key 做注意力打分，sigmoid 输出门控。略有改善但增加不少复杂度。
2. **仅注入早期层**：论文说早期层获益最多的相反——反而伤害了性能。模型似乎需要跨所有层统一注入。
3. **三元组**：扩展到同时哈希 2-gram 和 3-gram。无改善。
4. **仅双元 + x0 风格注入（modded-nanogpt engram-lite 方法）**：简单哈希 + 零初始化 embedding 表 + 逐层可学习 lambda + 每层残差注入。这种方法有效且持续改善。

### 最优超参数
- **表大小**：`vocab_size * 5`（32K 词表约 164K 条目）
- **注入方式**：每层通过可学习 `bigram_lambdas` 注入（初始化 0.1 优于 0.0）
- **Normalization**：embedding 上不 norm 略好
- **初始化**：零初始化 embedding 权重

### 核心经验
1. **门控在我们规模无帮助**——论文精细门控增加参数和复杂度，无改善。modded-nanogpt 也发现"简单直接加残差远超复杂方法"
2. **统一注入优于仅早期注入**——论文发现相反，但 x0 风格在 nanochat 更有效
3. **双元足够**——三元未帮助，额外上下文不抵稀释的容量
4. **规模很重要**——Engram 论文结果在 27B 参数 + MoE 上；在我们的约 100M-1B 规模，更简单的方法胜出

### 增加的参数量
d12 模型 `table_multiplier=5`：约 126M 参数（双元 embedding）+ 可忽略的 lambdas。现在大量参数在 embedding 表中：token embeddings、bigram embeddings、value embeddings。仅约四分之一参数是权重投影，绝大多数是 embedding 表。然而在所有轴（步数、wall clock、FLOPs）上，这种参数膨胀的架构确实击败了 baseline，成为新的默认。

加入 engram-lite 后重新运行了缩放定律来确定新的最优 tokens:params 比例。Kaplan 风格的比例最一致，约 **10.5** 成为新的默认值。

---

## 2026-01-19 to 2026-01-22：优化器超参数扫描

运行了约 320 次实验，从 d12 → d16 → d20 缩放，寻找最优优化器超参数。

### 核心发现
1. **超参数是规模相关的**：在 d12 上有效的并不会迁移到 d20。对 d12 有利的精细调优在 d20 上反而有损。
2. **改善幅度随规模缩小**：d12 上约 0.002 改善，d20 上约 0.0007。baseline 在更大模型上本就更优。
3. **存在急剧断崖**：x0_beta1=0.98 是灾难性的，而 0.96 是最优的。
4. **不要在小代理模型上过度调优**：发布前在目标规模上验证。

### 最终建议
生产 d20 运行只加 `--x0-lambdas-beta1=0.96`，跳过其他在小规模发现的所有调整。

---

## 2026-01-18：更多各种实验

- **Muon 自定义 kernel**（XXT 等）：目标测试中改善约 20%，但在实际训练中完全被噪声淹没（Muon 计算分散到所有 worker）。因复杂度膨胀放弃。
- **融合 QKV O Linear 层**：约零影响。
- **QKV 和 O 门控 `sa_lambdas`**：因 RMSNorm 的存在（消除任何标量乘数效果）有些困惑。帮助极微（约 1e-4 loss），为控制复杂度放弃。

---

## 2026-01-17：各种实验

modded-nanogpt 使用 [Value Embeddings](https://arxiv.org/abs/2410.17897)（VEs）配合有趣的 U 型结构。今天尝试了大量调整：

- VE 在每层/交替层/U 型/前后。交替层效果最好——比 modded-nanogpt 多得多
- 大量参数共享想法减少参数量的——全部失败
- 低秩分解、投影等减少参数量的——全部失败
- 门控有帮助

**总结：模型极度偏好 Value Embeddings。**这是一种以几乎零 FLOPs 代价增加大量容量（参数）的方法，因为这些 embedding 只是直接加到 Values 张量上。任何减少 VE 容量的尝试（参数共享、低秩、投影）都会失败。模型想要很多 VE、全部容量，这样做在所有轴上（步数、FLOPs、wall clock）都胜出。重新运行缩放定律发现由于模型参数极度膨胀，最优比例已从 8 减半到 4——远低于 Chinchilla 的 20。

**其他实验：**
- aspect_ratio=128 比 64 差——LLM 偏好更瘦更深的架构
- head_dim 明确偏好 128（更少更大头）

---

## 2026-01-17：Modded-nanogpt 想法扫描（续）

| 想法 | 结果 | 备注 |
|------|------|------|
| Attention gates | 无改善 | 每头可学习门控在注意力输出上。+1GB 内存，降低效率 |
| Batch size schedule | 放弃 | 8→16→24 配合 LR 缩放。训练脚本太臃肿复杂，不值认知负担 |
| Value embeddings | 显著有帮助 | 实验仍在进行 |

---

## 2026-01-16：Flash Attention 3 回退到 SDPA

为没有 Hopper GPU 的用户添加了从 Flash Attention 3 自动回退到 PyTorch `scaled_dot_product_attention`（SDPA）的支持。

### 实现
创建了 `nanochat/flash_attention.py`——一个统一接口：
- import 时检测 FA3 可用性（需要 sm90+ / Hopper）
- 导出与 FA3 API 完全匹配的 `flash_attn` 对象
- 根据硬件自动路由到 FA3 或 SDPA
- 处理张量布局差异：FA3 使用 (B, T, H, D)，SDPA 使用 (B, H, T, D)
- 通过显式 mask 为 SDPA 实现滑动窗口注意力
- 为 SDPA 手动管理 KV 缓存

### 关键变更
**gpt.py**：仅 import 行和注释改变。**engine.py**：零变更。**base_train.py**：添加状态输出和警告信息：是否使用 FA3 或 SDPA 回退；无 FA3 时效率损失；如果 `--window-pattern` 不是 "L" 的滑动窗口支持警告。

SDPA 回退比 FA3 显著更慢，尤其缺少滑动窗口注意力支持。使用 SDPA 回退时建议 `--window-pattern L`（全上下文）。

---

## 2026-01-16：Modded-nanogpt 想法扫描（大多负面）

测试了若干来自 modded-nanogpt 的架构想法，看是否适用于 nanochat。以下均无帮助：

| 想法 | 结果 | 备注 |
|------|------|------|
| Half-truncated RoPE | 无改善 | 仅前一半 head dims 获取 RoPE，后一半"静止" |
| Asymmetric softcap | 略差 | 可能仅对 FP8 有帮助 |
| Smear gate | 可忽略 | 通过可学习门控混合相邻 token。微改善不值新增参数量 |
| Backout | 无改善 | 约网络 60% 处保存激活，末尾减去缩放版 |
| Skip connection | 略差 | 约 25% 处保存，50% 处加入。额外 +2GB 内存 |

Value Embeddings 确实有前景，需要更细致的探索。

---

## 2026-01-15：Olmo 预训练混合（负面结果）

尝试用 Olmo 3 预训练数据集 [allenai/dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) 替代 FineWeb-edu。遇到很多[错误和问题](https://huggingface.co/datasets/allenai/dolma3_mix-6T/discussions/2)（下载和处理），注意到一些质量问题（如有些文档极端短，像"5"）。通过一些合理 hack 绕过（如拒绝少于 100 字符的文档），按 FineWeb 完全相同方式处理，重新训练分词器，训练 d16 模型。**CORE 得分从 15.5 降至 13.8——结果显著更差。**

**结论：负面结果。回退到 FineWeb-edu。**

---

## 2026-01-13：变长注意力 Varlen Attention（负面结果）

尝试用 Flash Attention 的 `flash_attn_varlen_func` 阻止注意力跨文档边界"泄漏"。假设 BOS 对齐数据加载器中多文档打包的边界会导致不必要的跨文档注意力。按 BOS 位置计算 cu_seqlens 实现了 varlen attention。遇到的问题：可变长度 cu_seqlens 导致 torch.compile 重编译（25s/iter！）→ 填充到固定大小修复；`nonzero()` 在编译模型内触发编译限制 → 搬到编译区域外修复。

**最终结果（d16）：** val_bpb 0.85427 vs 0.85407（baseline）。完全在噪声范围内。**不值得引入代码复杂度。合并到 master 不会做。**

---

## 2026-01-13：BOS 对齐数据加载器与装箱算法

重新设计了预训练数据加载器，确保每个序列以 BOS token 开头。

### BOS 对齐的三种方案
1. **Greedy-Crop BOS**：每行独立构建，文档填充，不完整的裁切。100% 利用率，39.4% token 被裁切丢弃
2. **Greedy-Pad**：不裁切但用 ignore token 填充空白位置。78% 利用率，浪费计算在 padding 上
3. **BestFit-Crop（新默认）**：缓存 N 个文档，贪婪选能完整放入最大的，找不到才裁切。100% 利用率，裁切率从 39.4% 降至 34.6%（约 12% 相对改善）

### 关键数据
T=2048：理论最低裁切率 22.9%（天然无法放入的长文档）。BestFit 额外裁切约 11.7%。轻微数据分布偏斜（长文档尾部 token 总是被丢弃），但不影响实际下游性能。BOS 对齐后验证 loss 显著下降，部分"虚假"（所有 token 都能看到 BOS 和完整文档上下文）。

---

## 2026-01-13：数字 Token 分割模式

验证了在 `SPLIT_PATTERN` 中使用 `\p{N}{1,2}` 的设计选择（之前只是猜测，留了 TODO）。GPT-4 用 `\p{N}{1,3}`。对于小词表（32K），实验证明 `{1,2}` 最优（val_bpb 0.965 vs {1,1} 0.969 vs {1,3} 0.972）。2 位数字分组是甜点——不太细粒度也不浪费稀少组合上。**保留 `{1,2}` 作为默认。**

---

## 2026-01-13：lm_head 的 FP8 训练

尝试在 lm_head 层使用 FP8 加速大词表投影 matmul。H100 的 FP8 tensor core 理论上有约 2× 加速。

### 尝试的方法
1. **动态缩放（失败）**：每次前向计算 `max(abs)` 作为 scale。`.item()` 调用导致 torch.compile 图断裂；用了 `@torch._dynamo.allow_in_graph` 但无加速；换成 `torch.library.custom_op` 后第一步优化器步后出现 NaN 梯度。根本原因是自定义 op、动态 scale 计算和 torch.compile 之间的交互脆弱。
2. **静态缩放（部分成功）**：借鉴 modded-nanogpt，预设 scale。正确工作，无 NaN，梯度正确。但奇怪的是 FP8 应节省内存（1 字节 vs 2 字节），实际却多用了 2GB。torch.compile 内部 kernel 可能产生额外缓冲。

### 结果（d12）
- **内存**：34 GB (BF16) → 36 GB (FP8) ❌
- **tok/sec**：baseline vs 约 1% 更快

**结论：lm_head 单独 FP8 不值得。**实现正常工作但提供微不足道的加速却**增加**了内存使用。未来还需研究 torchao 等库中相关内容。

---

## 2026-01-12：多 Token 预测 MTP

从 modded-nanogpt 移植多头预测。每个位置预测下 n 个 token，加权 loss。使用折叠 + gather + 交叉熵分解的批量计算。调度从 3-token 逐步退火到 1-token。

**结果（d12）：**
| Metric | Baseline | MTP |
|--------|----------|-----|
| GPU Memory | 34 GB | 47 GB ❌ |
| MFU | 41% | 40% |
| val/bpb (per step) | baseline | 相同/slightly worse |
| val/bpb (wall clock) | baseline | 明显更差 |

**结论：负面结果。**额外内存和计算开销不抵收益。辅助 loss 信号在其他场景（更大模型、不同架构？）可能有用，但对 nanochat 纯粹是开销。

---

## 2026-01-11：滑动窗口注意力

添加了可配置的滑动窗口注意力，受 GPT-3 交替短/长模式启发。`window_pattern` 字符串平铺到各层，最后一层强制 L（全上下文）。`SSSL` 效果很好（每第 4 层用全上下文）。计算节省显著。

---

## 2026-01-11：Flash Attention 3 集成

用 Flash Attention 3 替代了 PyTorch SDPA。通过 `kernels` 包从 HuggingFace Hub 加载预构建 wheel。FA3 采用 (B, T, H, D) 布局（匹配投影输出，无需转置）。GQA 自动处理。训练用 `flash_attn_func`，推理用 `flash_attn_with_kvcache` 一次调用处理所有缓存场景。

**结果：tok/sec 开箱即用约提升 9%。** 基准测试显示在真训尺寸（B=32, seq=2048）上 FA3 是 FA2 的 2× 快。

---

## 2026-01-11：逐层残差标量（x0 & resid lambdas）

从 modded-nanogpt 中移植了可学习逐层残差连接的想法。

### 变更内容
1. **x0_lambdas**：保存初始归一化 embedding 为 `x0`，每层把 x0 混入：`x = resid_lambdas[i]*x + x0_lambdas[i]*x0`。零初始化，提供从 embedding 到深层级的直接通道。
2. **resid_lambdas**：逐层残差乘性缩放。初始化为 1.0。

### 关键发现：不同 LR 敏感性
- **x0_lambdas（加法）**：可用正常 LR（约 0.5）。加 x0 的一小部分是宽容的。
- **resid_lambdas（乘法）**：需要约 100× 更小 LR（约 0.005）。乘法在跨层时复合。

### 实验结果
所有深度均有持续改善（d8 → d20：val bpb 改善约 0.004-0.01）。最优 LR 随深度变化，0.5 是合理默认值。零计算开销的稳健改善。

---

## 2026-01-10：Muon 优化器升级与谨慎权重衰减

从 NorMuon（modded-nanogpt）中挑选改进移植到本项目的简化版 Muon。

### 变更内容
1. **Polar Express 正交化**：替代 Newton-Schulz 迭代。新旧方法无明显差异，但保留 Polar Express 为默认。
2. **NorMuon 方差减小**：添加每神经元/列的自适应学习率。维护形状为 `[rows,1]` 或 `[1,cols]` 的二阶动量缓冲区，基于运行方差估计归一化更新。内存开销约 1/max(rows,cols) 每个参数，可忽略。非常小的改善，已启用并保持默认。
3. **谨慎权重衰减**：仅在 `update * weight >= 0` 时衰减权重。标准权重衰减总是向零拉，谨慎衰减在梯度已将权重推向零时跳过额外惩罚。稳健的改善，现默认开启。
4. **权重衰减调度**：添加了从 1.0 到 0.0 线性衰减的调度。实验证明优于静态设置。

### 权重衰减缩放定律
扫描 d8 → d20 的最优权重衰减，发现 **WD ∝ 1/width²**（幂律指数约 1.97）。实用公式：`WD_target = WD_ref × (d_ref/d_target)²`。

---

## 2026-01-08：梯度裁剪实验

假设梯度裁剪可能不必要，测试了 L2 范数裁剪和各种阈值。任何规模均无收益。代码自然产出行为良好的梯度。删除了所有 grad-clip 代码路径，提高了一些 MFU（不需要计算和同步梯度范数）。
