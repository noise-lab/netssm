# Code modified from: https://github.com/havenhq/mamba-chat/blob/main/train_mamba.py
"""Training script for NetSSM models.

This module provides functionality to train or fine-tune NetSSM Mamba models
on tokenized network traffic datasets. It supports resuming from checkpoints
and saving model weights at specified epochs.

Usage:
    uv run train.py --model <config_path> --tokenizer <tokenizer_path> \\
        --data_path <dataset_path> --output <output_dir> --num_epochs <epochs>
"""

import argparse
import glob
import os

import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from mamba_ssm.utils.hf import load_config_hf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from models.config_mamba import MambaConfig
from models.mixer_seq_simple import MambaLMHeadModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MambaTrainer(Trainer):
    """Custom HuggingFace Trainer for Mamba models.

    Extends the standard Trainer to implement custom loss computation
    for autoregressive language modeling with Mamba architecture.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        return lm_loss

    def save_model(
        self, output_dir, num_epochs=None, epoch_save=False, _internal_call=None
    ):
        if epoch_save:
            save_dir = output_dir + "/" + str(num_epochs) + "_epochs"
            self.model.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(output_dir)


def get_checkpoint(model):
    """Find the most recent checkpoint in a model directory.

    Args:
        model: Path to the model output directory.

    Returns:
        Path to the most recent checkpoint, or None if no checkpoints exist.
    """
    cwd = os.getcwd()
    directory = os.path.join(cwd, model)
    ckpts = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    for c in ckpts:
        if "epoch" in c:
            ckpts.remove(c)
    if not ckpts:
        return None
    load_ckpt = max(ckpts, key=lambda d: os.path.getmtime(os.path.join(directory, d)))
    ckpt = model + "/" + load_ckpt
    return ckpt


def get_dataset(data_path, tokenizer):
    """Load and prepare a dataset for training.

    Args:
        data_path: Path to either a .jsonl file or directory containing .arrow files.
        tokenizer: Tokenizer to apply to raw JSONL data.

    Returns:
        HuggingFace Dataset ready for training.
    """
    if ".jsonl" in data_path:
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.map(
            lambda e: tokenizer(e["Data"], padding="longest"), batched=True
        )
        dataset = dataset.with_format("torch")
    else:
        find_arrow_files = lambda directory: glob.glob(
            os.path.join(directory, "*.arrow")
        )
        arrow_files = find_arrow_files(data_path)
        arrow_files.sort()
        dataset = concatenate_datasets(
            [Dataset.from_file(arrow_file) for arrow_file in arrow_files]
        )
    return dataset


def parse_dtype(s):
    try:
        return getattr(torch, s)
    except AttributeError:
        raise argparse.ArgumentTypeError(f"Invalid dtype: {s}")


def run(args):
    """Main training loop.

    Args:
        args: Parsed command-line arguments containing training configuration.
    """
    transformers.logging.set_verbosity_info()
    resume = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    try:
        load_ckpt = get_checkpoint(args.output)
        model = MambaLMHeadModel.from_pretrained(
            load_ckpt, dtype=args.torch_dtype, device="cuda"
        )
        resume = True
    except:
        config_data = load_config_hf(args.model)
        config_data["vocab_size"] = len(tokenizer)
        config = MambaConfig(**config_data)
        model = MambaLMHeadModel(config, dtype=args.torch_dtype, device="cuda")

    dataset = get_dataset(args.data_path, tokenizer)

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=args.output,
            save_total_limit=2,
            logging_steps=50,
            save_steps=500,
            num_train_epochs=args.num_epochs,
        ),
    )
    if resume:
        trainer.train(get_checkpoint(args.output))
    else:
        trainer.train()

    trainer.save_model(args.output, args.num_epochs, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/configs/default")
    parser.add_argument("--output", type=str, default="checkpoints/toy_example")
    parser.add_argument(
        "--tokenizer", type=str, default="tokenizers/mia_netssm_nprint_os_tok"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--torch_dtype",
        type=parse_dtype,
        default="bfloat16",
        help="Torch dtype (e.g., float32, float16, bfloat16)",
    )
    args = parser.parse_args()
    run(args)
