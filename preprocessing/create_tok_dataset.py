import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=4,
        help="Number of inputs to process at a time.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        required=False,
        default=4,
        help="Number of processes to spawn for tokenization.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=0,
        help="Length to truncate all sequences to (used to handle for memory constraints on GPU).",
    )
    parser.add_argument(
        "--padding",
        required=False,
        action="store_true",
        help="If used, will pad inputs to the length of the longest sequence in the batch.",
    )
    requiredNamed = parser.add_argument_group("required arguments")
    requiredNamed.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer."
    )
    requiredNamed.add_argument("--data_path", type=str, required=True)
    requiredNamed.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load dataset in native Python format
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # Process the dataset in smaller chunks, if needed, to avoid memory overflow
    def process_batch(batch):
        tokenized_data = None
        # Tokenize the 'Data' field in each record
        if args.max_len != 0:
            tokenized_data = tokenizer(
                batch["Data"], max_length=args.max_len, truncation=True
            )
        else:
            tokenized_data = tokenizer(batch["Data"], padding="max_length")
        return tokenized_data

    # Apply tokenization using the map function with smaller batch sizes
    dataset = dataset.map(
        process_batch, batched=True, batch_size=args.batch_size, num_proc=args.num_proc
    )

    # Save the processed dataset to disk
    dataset.save_to_disk(args.out_path)
