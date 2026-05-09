"""
Fineweb-edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""
import os

# !! Redirect HF cache to DATA disk BEFORE importing datasets !!
HF_CACHE_DIR = "/media/mendax3301/DATA/hf/cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR

import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# --------------------------------------------------------
local_dir = "/media/mendax3301/DATA/hf/edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard

DATA_CACHE_DIR = local_dir
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token values must fit in uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def shard_exists(shard_index):
    """Check if a shard has already been written (resume support)."""
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
    return os.path.exists(filename)

def count_completed_tokens(shard_index):
    """Count how many tokens are already saved across completed shards."""
    return shard_index * shard_size

# tokenize all documents and write output shards
nproc = max(1, os.cpu_count() // 2)

# Find the last completed shard to resume from
shard_index = 0
while shard_exists(shard_index):
    print(f"Shard {shard_index} already exists, skipping...")
    shard_index += 1

tokens_to_skip = count_completed_tokens(shard_index)
print(f"Resuming from shard {shard_index} (skipping {tokens_to_skip:,} tokens)")

with mp.Pool(nproc) as pool:
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    tokens_count = 0
    progress_bar = None
    tokens_seen = 0  # total tokens seen across all docs

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        tokens_seen += len(tokens)

        # Skip docs that were already written in completed shards
        if tokens_seen <= tokens_to_skip:
            continue

        if tokens_count + len(tokens) < shard_size:
            all_tokens_np[tokens_count:tokens_count + len(tokens)] = tokens
            tokens_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            remainder = shard_size - tokens_count
            progress_bar.update(remainder)
            all_tokens_np[tokens_count:tokens_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
            tokens_count = len(tokens) - remainder

    # Write the last partial shard
    if tokens_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:tokens_count])
        print(f"Wrote final shard {shard_index}")