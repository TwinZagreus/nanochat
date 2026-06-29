"""
Minimal SFT that only uses local identity_conversations.jsonl.
No HuggingFace dependency. Use this when HF is unreachable.
极简SFT: 仅用本地 identity_conversations.jsonl (1000条身份对话)，不依赖HuggingFace。
适用于: 国内网络无法访问HF、快速验证SFT流程、小规模实验。
完整SFT请用 chat_sft.py (需要SmolTalk/MMLU/GSM8K等HF数据集)

Run as: python -m scripts.chat_sft_mini
流程: 加载预训练模型 → 用identity数据做SFT → 保存到 chatsft_checkpoints/
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import argparse
from nanochat.common import compute_init, print0, get_base_dir, autodetect_device_type, preflight_compile_check
preflight_compile_check()

parser = argparse.ArgumentParser()
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--eval-every", type=int, default=200)
parser.add_argument("--run", type=str, default="dummy")
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print0(f"Device: {device}")

from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from tasks.customjson import CustomJSON

# Load base model
model, tokenizer, meta = load_model("base", device, phase="train")
print0(f"Loaded base model d{model.config.n_layer}, step {meta['step']}")

# 身份数据路径: 需先通过 curl 从 S3 下载 (不用翻墙)
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
base_dir = get_base_dir()
identity_path = os.path.join(base_dir, "identity_conversations.jsonl")
assert os.path.exists(identity_path), f"Not found: {identity_path}. Download with: curl -L -o {identity_path} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
train_dataset = CustomJSON(filepath=identity_path)
print0(f"Training data: {len(train_dataset)} conversations")

# Optimizer
optimizer = model.setup_optimizer(unembedding_lr=args.lr, embedding_lr=args.lr, matrix_lr=args.lr, weight_decay=0.0)

# Data generator (same logic as chat_sft.py)
row_capacity = args.max_seq_len + 1
bos_token = tokenizer.get_bos_token_id()

def data_generator():
    conv_buffer = []
    idx = 0
    while True:
        for _ in range(args.device_batch_size):
            row, mask_row = [], []
            while len(row) < row_capacity:
                if not conv_buffer:
                    conv_buffer = [tokenizer.render_conversation(train_dataset[(idx + i) % len(train_dataset)]) for i in range(32)]
                    idx += 32
                ids, mask = conv_buffer.pop(0)
                if len(ids) <= row_capacity - len(row):
                    row.extend(ids)
                    mask_row.extend(mask)
                else:
                    conv_buffer.insert(0, (ids, mask))
                    remaining = row_capacity - len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    break
            yield row[:row_capacity], mask_row[:row_capacity]

def sft_loader():
    rows, mask_rows = [], []
    gen = data_generator()
    while True:
        for _ in range(args.device_batch_size):
            r, m = next(gen)
            rows.append(r)
            mask_rows.append(m)
        batch = torch.tensor(rows, dtype=torch.long)
        inputs = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device).clone()
        mask = torch.tensor(mask_rows, dtype=torch.int8)
        targets[mask[:, 1:] == 0] = -1
        rows.clear()
        mask_rows.clear()
        yield inputs, targets

loader = sft_loader()
x, y = next(loader)

# Training
smooth_loss = 0
step = 0
while step < args.num_iterations:
    t0 = time.time()
    model.train()
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    x, y = next(loader)

    smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()
    dt = time.time() - t0
    step += 1
    if step % 10 == 0:
        print0(f"step {step:05d}/{args.num_iterations:05d} | loss: {smooth_loss/(1-0.9**step):.6f} | dt: {dt*1000:.1f}ms")

# Save
from nanochat.checkpoint_manager import save_checkpoint
checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", f"d{model.config.n_layer}")
save_checkpoint(checkpoint_dir, step, model.state_dict(), None, {
    "step": step, "model_config": meta["model_config"]
})
print0(f"Saved to {checkpoint_dir}")
print0("Done! Now run: python -m scripts.chat_cli")
