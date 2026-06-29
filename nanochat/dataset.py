"""
The base/pretraining dataset is a set of parquet files.
预训练数据集: ClimbMix-400B (HuggingFace S3托管)，parquet格式存储。

This file contains utilities for:
- 遍历parquet文件流式读取文档
- 按需从HF/S3下载分片 (多进程并行，指数退避重试)

For details of how the dataset was prepared, see `repackage_data_reference.py`.
每个分片 ~100MB 压缩文本，共6542个训练分片 + 1个验证分片(shard_06542)
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# 当前预训练数据集的具体配置
# The specifics of the current pretraining dataset

# 数据集的远程托管地址，按需下载
# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # 最后一个数据分片是 shard_06542.parquet / the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # 文件名格式 / format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# 以下函数为其他模块提供的工具方法，可被导入使用
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """
    扫描数据目录，返回所有 parquet 文件的完整路径列表。
    Looks into a data dir and returns full paths to all parquet files.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # 旧版兼容代码：从 FinewebEdu-100B 升级到 ClimbMix-400B 的过渡逻辑，后续将删除
    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # 回退到旧版数据目录 (FinewebEdu-100B)
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    批量迭代数据集，按底层 row_group 批次读取以提高效率。
    - split: "train" 或 "val"，最后一个 parquet 文件固定为验证集
    - start/step: 用于 DDP 多卡训练时跳过行，如 start=rank, step=world_size
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    # 训练集用除最后一个外的所有分片，验证集只用最后一个分片
    # Training set uses all shards except the last; validation set uses only the last shard
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        # 打开 Parquet 文件，仅读取元数据（不加载全部数据到内存）
        # Parquet 是列式存储格式，文件内部按 row_group 分块组织
        # Open the Parquet file (metadata only, does not load all data into memory)
        # Parquet is a columnar format; internally organized into row_groups
        pf = pq.ParquetFile(filepath)
        # 按 row_group 逐块遍历，每次只读取一个 row_group，避免一次性加载整个文件
        # start/step 用于 DDP 多卡训练时分片跳过（如 rank=0 读 0,2,4... rank=1 读 1,3,5...）
        # Iterate row_group by row_group to avoid loading the entire file at once
        # start/step enable DDP sharding (e.g. rank=0 reads 0,2,4... rank=1 reads 1,3,5...)
        for rg_idx in range(start, pf.num_row_groups, step):
            # 读取一个 row_group 到内存，返回一个 Table 对象
            # Read one row_group into memory, returns a Table object
            rg = pf.read_row_group(rg_idx)
            # 提取 'text' 列并转为 Python 列表，每个元素是一个文档的完整文本
            # Extract the 'text' column and convert to a Python list; each element is a full document text
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """
    下载单个数据分片文件，带指数退避重试机制。
    Downloads a single file index, with some backoff.
    """

    # 构建本地文件路径，若已存在则跳过
    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # 构建远程下载URL
    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # 带重试的下载逻辑
    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # 先写入临时文件，完成后重命名为正式文件
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB 块大小 / 1MB chunks
                    if chunk:
                        f.write(chunk)
            # 将临时文件移动到最终位置
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理所有不完整的文件
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 指数退避重试：等待 2^attempt 秒
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载预训练数据集分片 / Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="要下载的训练分片数，-1 表示禁用 / Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="并行下载的进程数 / Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # 准备输出目录
    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # 用户通过 -n 指定要下载的训练分片数量，验证分片始终会被下载（固定为最后一个分片）
    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # 始终下载验证分片 / always download the validation shard

    # 多进程并行下载分片
    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # 输出下载结果统计
    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
