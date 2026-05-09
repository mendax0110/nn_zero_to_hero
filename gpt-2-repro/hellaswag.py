"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import os
import json
import requests
import argparse
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    total_size = int(response.headers.get('content-length', 0))
    with open(fname, 'wb') as f, tqdm(
        desc=f"Downloading {fname}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
            
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Download the HellaSwag dataset for a given split (train, val, or test)."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    url = hellaswags[split]
    fname = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(fname):
        download_file(url, fname)
    else:
        print(f"{fname} already exists, skipping download.")
        
def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    
    # data needed to repoduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": []
    }
    
    # gather up all the tokesn
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # add a space before the ending, since the context doesn't end with a space
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)
        
    # have to be careful during the collation because the num of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.bool)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
        
    return data, tokens, mask, label

def iterate_examples(split):
    # there are 10042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example
            
@torch.no_grad()
def evaluate(mode_type, device):
    
    torch.set_float32_matmul_precision('high') # works only on modern GPU's
    model = GPT2LMHeadModel.from_pretrained(mode_type)
    model.to(device)
    model = torch.compile(model)
    
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        # get the logits
        logits = model(tokens).logits
        # eval the autoregressive loss at all positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_labels = shift_labels.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_labels, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the AVG loss just for the completion region (where mask is 1) in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token, and end at the last completion token
        masked_shift_losses = shift_losses * shift_mask
        # SUM and DIV by the num of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        
        # accum stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"example {num_total}: pred {pred} (norm {pred_norm}), label {label}, acc {num_correct/num_total:.4f}, acc_norm {num_correct_norm/num_total:.4f}")
        
        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print(f"example {num_total}:")
            print(f"  ctx: {data['ctx_tokens']}")
            for i, end in enumerate(data["ending_tokens"]):
                print(f"  ending {i}: {end}")
                print(f"    sum_loss: {sum_loss[i].item():.4f}, avg_loss: {avg_loss[i].item():.4f}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gpt2", help="which model to evaluate (gpt2 or gpt2-xl)")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="which device to run on")
    args = parser.parse_args()
    
    evaluate(args.model, args.device)