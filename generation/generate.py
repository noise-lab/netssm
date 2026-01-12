# Copyright (c) 2023, Tri Dao, Albert Gu.
"""Token generation script for trained NetSSM models.

This module provides functionality to generate raw token sequences from
a trained NetSSM model. The generated tokens represent network packet
data and can be converted to PCAP format using conversion.py.

Usage:
    uv run generation/generate.py --model <checkpoint_path> \\
        --tokenizer <tokenizer_path> --prompt "<|netflix|>" --genlen 1000
"""

import os
import sys
import argparse
import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

CWDPATH = os.getcwd().split('/')[-1]
if CWDPATH == "generation":
    sys.path.append(os.path.dirname(os.getcwd()))
elif CWDPATH == "netssm":
    sys.path.append(os.getcwd())
 
from models.mixer_seq_simple import MambaLMHeadModel

def parse_dtype(s):
    """Parse a string to a PyTorch dtype.

    Args:
        s: String representation of dtype (e.g., 'float32', 'bfloat16').

    Returns:
        Corresponding torch dtype.

    Raises:
        ArgumentTypeError: If the dtype string is invalid.
    """
    try:
        return getattr(torch, s)
    except AttributeError:
        raise argparse.ArgumentTypeError(f"Invalid dtype: {s}")


def write_model_output(model_name, decoded_tokens, out_dir):
    """Write generated tokens to a text file.

    Args:
        model_name: Name of the model (unused, kept for compatibility).
        decoded_tokens: String of decoded token output.
        out_dir: Directory to write the output file.
    """
    with open(out_dir + "/" + "generated.txt", 'w') as f:
        f.write(decoded_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--experiment", type=str, default="RUN_1")
    parser.add_argument("--experiment_base_dir", type=str, default="EXP_1")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--minp", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--torch_dtype", type=parse_dtype, default="bfloat16", help="Torch dtype (e.g., float32, float16, bfloat16)")
    parser.add_argument("--gen_len_pkts", action='store_true', help="Pass to generate packets instead of explicit generation length.")
    args = parser.parse_args()

    device = "cuda"
    dtype = args.torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()
    print(f"Loading model {args.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + args.genlen
    torch.cuda.synchronize()
    start = time.time()
    fn = lambda: model.generate(
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
        gen_len_pkts=args.gen_len_pkts
    )
    out = fn()

    torch.cuda.synchronize()
    print(f"{args.model} prompt processing + decoding time: {(time.time() - start) / 1*1000:.0f}ms")
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
    generated = tokenizer.batch_decode(out.sequences.tolist())[0]
    if CWDPATH == "generation":
        out_dir = os.path.join("..", "inference", args.experiment_base_dir, args.experiment, "")
    elif CWDPATH == "netssm":
        out_dir = os.path.join("inference", args.experiment_base_dir, args.experiment, "")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Attempting to write to {out_dir}...")
    write_model_output(args.model, generated, out_dir)
