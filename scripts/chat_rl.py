"""
在 GSM8K 上通过 "GRPO" 进行强化学习。
Reinforcement learning on GSM8K via "GRPO".

我给 GRPO 加引号是因为我们实际上得到了一个简单得多、
更接近纯 REINFORCE 的方法：
I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) 删除信任域，因此没有对参考模型的 KL 正则化
1) Delete trust region, so there is no KL regularization to a reference model
2) 我们在策略上(on-policy)运行，因此不需要 PPO 的 ratio+clip。
2) We are on policy, so there's no need for PPO ratio+clip.
3) 使用 DAPO 风格的归一化，是 token 级别而非序列级别。
3) We use DAPO style normalization that is token-level, not sequence-level.
4) 与 z-score 归一化 (r - mu)/sigma 不同，只使用 (r - mu) 作为优势值。
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

单 GPU 运行：
1 GPU:
python -m scripts.chat_rl

8 GPU 运行：
8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# 命令行参数
# CLI arguments
parser = argparse.ArgumentParser(description="在 GSM8K 上进行强化学习 / Reinforcement learning on GSM8K")
# 日志记录
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' 表示禁用 wandb 日志） / wandb run name ('dummy' disables wandb logging)")
# 运行时
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps（留空则自动检测） / cuda|cpu|mps (empty = autodetect)")
# 模型加载
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="要加载的模型标签 / model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="要加载的模型步数 / model step to load from")
# 训练周期
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="在 GSM8K 上训练的 epoch 数 / number of epochs over GSM8K")
# 批量大小 / 采样
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="每次前向传播的最大批量大小 / max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="每优化步在所有 rank 上的总示例数 / total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="每个示例/问题的采样数 / number of samples per example/question")
# 生成参数
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="每个样本最多生成的 token 数 / max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="采样温度 / sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k 采样（0 = 禁用） / top-k sampling (0 = disabled)")
# 优化参数
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="embedding 参数的学习率 (Adam) / learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="unembedding 参数的学习率 (Adam) / learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数的学习率 (Muon) / learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="embedding/unembedding 参数的权重衰减 (Adam) / weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="初始学习率占基础学习率的比例 / initial LR as fraction of base LR")
# 评估 / 检查点
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="每 N 步评估 pass@k / evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="pass@k 评估的示例数 / number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="每 N 步保存检查点 / save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# 初始化计算/精度
# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # 该进程负责日志记录、保存检查点等 / this process will do logging, checkpointing etc.

# wandb 日志初始化
# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# 初始化模型和分词器
# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # 用于采样 rollout / for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / 采样生成器循环，产生批量训练示例
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    """
    训练数据的生成器函数。循环遍历训练集，对每个示例采样多个补全，计算奖励和优势值。
    Generator function for training data. Cycles through the training set, samples multiple completions per example, computes rewards and advantages.
    返回 inputs/targets (B, T) 和 rewards/advantages (B,) 用于策略梯度更新。
    Returns inputs/targets (B, T) and rewards/advantages (B,) for policy gradient updates.
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # 可以使用此 token，仅用于 padding，不参与 loss 计算 / ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # 每个 rank 负责训练数据中的不同示例 / each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # 首先获取包含用户和助手消息的完整对话
        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # 将对话 token 化，删除最后一条 Assistant 消息，改为让 Assistant 进行补全
        # （即保留 <|assistant_start|>，但删除其后的所有内容）
        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # 使用批量生成产生 num_samples 个样本，用循环避免 OOM
        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # 确保模型处于评估模式 / ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # 顺序执行以防止 OOM / go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # int32 正半部分 / positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # 必须确保每次采样步骤的 seed 不同 / must make sure to change the seed for each sampling step
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # 计算每个样本的奖励
        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 获取仅生成部分的 token（prompt 之后）
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # 解码生成的回复
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # 计算奖励
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # 对序列进行 padding，使它们的长度（时间维度）对齐
        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # 将序列和掩码堆叠为 PyTorch tensor
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # 生成 Transformer 的自回归输入和目标
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # 克隆以避免原地修改 / clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- 此处为原地修改。-1 是忽略索引 / <-- inplace modification right here. -1 is the ignore index
        # 注意：Engine 对 prompt token 和 tool use token 都返回 mask=0。
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # 因此我们将（正确地）不会在 prompt token 或 tool use 强制 token 上训练。
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # 通过简单减去均值来计算优势值（而非 z-score 归一化的 (x-mu)/sigma）
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # 产出 inputs/targets 形状为 (B, T)，rewards 形状为 (B,)
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# GSM8K pass@k 的简单评估循环
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    评估 GSM8K 任务，返回评估结果记录的列表。
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    在分布式环境中，所有 rank 会协作，但此函数不会跨 rank 做汇总。汇总由调用方负责。
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    由于评估可能需要较长时间，此函数会逐条产出记录。
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # 使用 Engine 内置的批量生成，生成 k 个样本
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # 通常这是满足的。如果不满足可以加循环... / usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # 检查每个样本的正确性
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # 结构略臃肿，因为之前想做更复杂的日志记录。
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# 训练循环
# Training loop

# 初始化优化器
# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# 将初始学习率设置为基础学习率的一定比例
# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# 学习率调度器：简单的线性衰减到零，覆盖 num_steps
# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# 计算每个 rank 需要处理的示例数，以达到所需的 examples_per_step
# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # 每步的总 batch size（序列数） / total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # 每个 GPU 的示例数 / per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# 启动训练循环
# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # 定期评估模型并记录到 wandb
    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k，k=1..device_batch_size / pass@k for k=1..device_batch_size
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # 收集所有记录 / collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # 按总记录数归一化 / normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # 对数据集中多个示例的 rollout 进行前向/反向传播
    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # 获取一个 batch，对应训练数据集中的一个示例
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # 计算损失和梯度
        # Evaluate the loss and gradients
        model.train() # 确保模型处于训练模式 / ensure the model is in train mode
        # 需要再来一层循环，因为不能超过 device_batch_size
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # 取出本轮的 batch
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # 计算对数概率。注意 loss 计算的是 NLL = -logp，所以取反
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # 计算策略梯度目标。注意 ignore_index=-1 确保无效 token 的 loss 为 0。
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # 按有效 token 数、pass 数和 examples_per_rank 进行归一化
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # 注意，不需要 PPO 的 ratio+clip，因为我们是 on-policy
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # 最后，构建要最小化的 loss（而非最大化的目标）
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # 用于日志记录
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # 大量日志记录，反映本步 rollout 的情况
    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # 在所有 rank 之间汇总 / aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # 更新模型参数
    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master 进程定期保存模型。跳过第一步。保存最后一步。
    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # 基于基础模型的深度来命名模型标签 / base the model tag on the depth of the base model
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # 稍有不妥，利用 GPTConfig 的简洁性，TODO 改进 / slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # 注意：我们不费心保存优化器状态 / note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# 记录到报告
# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI 参数 / CLI args
])

wandb_run.finish() # wandb 运行结束 / wandb run finish
compute_cleanup()
