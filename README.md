# NetSSM: Multi-Flow and State-Aware Network Trace Generation using State-Space Models

This repository contains the code and datasets for our paper ["NetSSM: Multi-Flow and State-Aware Network Trace Generation using State-Space Models"](https://arxiv.org/pdf/2503.22663).

* [Requirements](#requirements)
* [Datasets](#datasets)
* [Google Colab Toy Example](#google-colab-toy-example)
* [Running the NetSSM pipeline from scratch](#running-the-netssm-pipeline-from-scratch)
  * [Setup](#setup)
  * [Dataset preparation](#dataset-preparation)
  * [Training/Finetuning](#training-and-finetuning)
  * [Generation](#generation)
* [Citation](#citation)

## Requirements

  * [uv Python package manager](https://github.com/astral-sh/uv)
  * Go >= v1.21.4
  * Python >= 3.10
  * libpcap

## Datasets

PCAP datasets corresponding to Table 1 from the paper can be found at the below links:
* Multimedia Traffic:
  * [Video Streaming](https://drive.google.com/file/d/1QsEOblt36uKY3qk0yDCMcxnqnXP1Jwpm/view?usp=share_link)
  * [Video Conferencing](https://drive.google.com/file/d/1_2t5lxzLNVAOg7MfQrWFB7coiAv-ZlAM/view?usp=share_link)
  * [Social Media](https://drive.google.com/file/d/1d4x6uvFZmNhzy_XR_f6SSmnGIz953tbM/view?usp=share_link)
* [Netflix Streaming](https://drive.google.com/file/d/1azHb6dDFGiHiIQAy4RixEMCe0KmuBKWx/view?usp=share_link)
* [YouTube Streaming](https://github.com/Wimnet/RequetDataSet/tree/master)

Please cite the respective dataset source in the folder README/repo if you use a dataset. Table 1 from the paper also provides a citation.

## Google Colab Toy Example

The Jupyter notebook at [`example/train_netssm_from_scratch.ipynb`](https://github.com/noise-lab/netssm/blob/main/example/train_netssm_from_scratch.ipynb) contains a thorough walkthrough of setting up dependencies, training NetSSM from scratch/resuming from a checkpoint, and generating synthetic PCAPs. We recommend opening the notebook in Google Colab. If run on one of the default Colab T4 GPUs, the training and generation steps of the notebook may take a few minutes. 

## Running the NetSSM pipeline from scratch
### Setup

Install most dependencies:
```bash
# This will install most required Python dependencies
uv sync
```

The remaining setup is for `mamba-ssm` and `causal-conv1d`, which can be most easily installed by installing the wheels directly. Run the following two commands, and note the output:

`nvcc --version` or `/usr/local/cuda/bin/nvcc --version`: Note the CUDA compile release number.

`.venv/bin/python -c  "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"`: Note if `True` or `False` prints.

Download the corresponding wheel file for both [`mamba-ssm`](https://github.com/state-spaces/mamba/releases/tag/v2.2.6.post3) and [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.4), matching the naming scheme of the asset for your respective setup. Note, the previous `uv sync` command configures PyTorch 2.7 , so you shoud look for the wheel files that match this version, e.g.,:
```bash
# Downloads mamba-ssm and causal-conv1d for CUDA compiler version 12.X, PyTorch 2.7, abi == FALSE, Python version 3.10.X
wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Next, install the downloaded wheels with `uv`:
```bash
# Replace names with your downloaded files
uv pip install mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install causal_conv1d-1.5.4+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Dataset preparation

Raw PCAPs must be pre-processed into a string representation sequence of packets and then tokenized, for use with NetSSM.

#### Data pre-processing

Setup the preprocessor:
```bash
cd preprocessing
go mod init netssm_preprocessor
go mod tidy
go build
```

This builds the preprocessor binary called `netssm_preprocessor`. The preprocessor has the following usage options:

```
Usage of ./netssm_preprocessor:
  -in-csv string
        CSV file with PCAP paths (default "./")
  -in-dir string
        Directory with PCAPs (default "./")
  -label string
        Blanket label to use, if all PCAPs are of same service/type
  -label-csv in-dir
        CSV mapping pcaps in in-dir to their corresp. label (default "./")
  -out string
        Output JSONL dataset name (default "./")
  -truncate int
        OPTIONAL -- Length to truncate all samples to (default -1)
```

One of `-in-csv` or `-in-dir` must be provided. Similarly, either `label` or `-label-csv` must be provided. If `-in-csv` is used, the CSV pointed to should have header `File`, where the value in this column should be the full path to each PCAP. The CSV file for `-label-csv` should have the header `File,Label`, where the value in `File` should be only the filename of each PCAP. See `example/input/labels.csv` for an example of the labels file.

The preprocessor will parse raw PCAP files into a string of the raw bytes of comprising each packet in a capture. This string is prepended with the traffic label/type, and each packet is delimited by a `<|pkt|>` special token. For example: `<|netflix|> 226 70 154 78 108 47... <|pkt|> 00 128 153 192 8... <|pkt|>`.

#### Tokenizers

NetSSM's tokenizer maps the raw byte decimal values in $[0, 255]$ 1:1 to corresponding tokens. Both training from scratch or fine-tuning on new data will likely require a new custom tokenizer to handle this data. Use `tokenizers/create_tokenizer.py` to create this.

```bash
uv run create_tokenizer.py [-h] [--special_tokens SPECIAL_TOKENS] --tokenizer_name TOKENIZER_NAME

optional arguments:
  -h, --help            show this help message and exit
  --special_tokens SPECIAL_TOKENS
                        Special tokens to include in the tokenizer, input as space-delimitted string (e.g., "netflix facebook amazon").

required arguments:
  --tokenizer_name TOKENIZER_NAME
                        Name to save the tokenizer.
```

The output of this script is a folder containing various files that comprise a custom tokenizer.

#### Tokenizing Dataset

We tokenize the dataset produced by the pre-processing step before training, to more efficiently use GPU allocation. The script for this is at `preprocessing/create_tok_dataset.py`.

```bash
uv run create_tok_dataset.py [-h] [--batch_size BATCH_SIZE] [--num_proc NUM_PROC] [--max_len MAX_LEN] [--padding] --tokenizer TOKENIZER --data_path DATA_PATH --out_path OUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of inputs to process at a time.
  --num_proc NUM_PROC   Number of processes to spawn for tokenization.
  --max_len MAX_LEN     Length to truncate all sequences to (used to handle for memory constraints on GPU).
  --padding             If used, will pad inputs to the length of the longest sequence in the batch.

required arguments:
  --tokenizer TOKENIZER
                        Path to tokenizer.
  --data_path DATA_PATH
  --out_path OUT_PATH
```

The output of this script is a folder containing `.arrow` files of the tokenized representations of the dataset created in [Data pre-preprocessing](#data-pre-processing).

#### Summary

Putting it all together, the following code block explains how to convert a directory of PCAPs to their tokenized representations, using the `-label-csv` option for label mapping:

```bash
# 1. Parse PCAPs to string representations
./netssm_preprocessor -in-dir <PATH_TO_PCAP_DIR> -label-csv <PATH_TO_LABEL_CSV> -out <PATH_TO_DATASET_JSONL>
# 2. Create a new tokenizer including traffic labels
uv run create_tokenizer.py --special_tokens "netflix facebook amazon" --tokenizer_name <CUSTOM_TOKENIZER_OUTPUT_PATH>
# 3. Apply tokenizer to the raw dataset, for use in model training
uv run create_tok_dataset.py --max_len 100000 --padding --tokenizer <CUSTOM_TOKENIZER_OUTPUT_PATH> --data_path <PATH_TO_DATASET_JSONL> --out_path <DATASET_OUT_PATH>
```

The resulting tokenized dataset at `<DATASET_OUT_PATH>` is now usable for training/fine-tuning.

### Training and Finetuning

⚠️ **NetSSM (and as source the original Mamba/2) currently does not support multi-GPU training out of the box. See this issue ([#84](https://github.com/state-spaces/mamba/issues/84)) for some potential workarounds (untested for this repo).** ⚠️

After creating a dataset and tokenizer, and tokenizing the dataset using the above steps, training can be run using the `train.py` script in the root directory:

```bash
uv run train.py \
  --model=<PATH_TO_MODEL_CONFIG>
  --output=<PATH_TO_OUTPUT_FOLDER> \
  --data_path=<DATASET_OUT_PATH> \
  --tokenizer=<CUSTOM_TOKENIZER_OUTPUT_PATH> \
  --num_epochs=<NUM_EPOCHS>
```

`<PATH_TO_MODEL_CONFIG>` should resemble a file similar to that at `checkpoints/configs/default`, which specifies the model parameters.

If a prior model training checkpoint exists at `<PATH_TO_OUTPUT_FOLDER>`, the training script will automatically load from this checkpoint, and resume training. Otherwise, the model will begin training from scratch.

There are additional training parameters that can be passed to `train.py`; see the arguments in the file itself.

#### Fine-tuning

If you are using an existing model checkpoint, but want to fine-tune on new data with labels that do not exist in the checkpointed model/tokenizer, follow the steps in [Dataset preparation](#dataset-preparation), creating a new tokenizer that contains the new labels for the new data, and creating the tokenized dataset. Then when using `train.py`, set `--output` to the directory containing the starting checkpoint for fine-tuning, `--tokenizer` to the new tokenizer, and `--data_path` to the tokenized dataset.

### Generation

Generation runs in two steps: (1) generating the raw tokens corresponding to bytes in a PCAP and (2) converting these tokens to the actual trace.

#### Example with a pre-trained checkpoint

We provide a toy pre-trained checkpoint that can be used for generation out of the box. Download the pre-trained checkpoint:

```bash
source venv/bin/activate
gdown 1koMbDyaTi0buF1eoDplqOFtJLX-ssS6a
mv netflix_multi_100k_30_epochs.zip ./checkpoints && cd ./checkpoints && unzip netflix_multi_100k_30_epochs.zip && mv checkpoint-176460 netflix_multi_100k_30_epochs && cd ..
```

Then follow the steps for (1) and (2) below.

#### 1. Raw token generation

Example usage prompting to generate a PCAP of 1,000 packets of Netflix traffic using the pre-trained model checkpoint and corresponding multi-flow tokenizer.

```bash
uv run ./generation/generate.py \
  --prompt "<|netflix|>" \
  --model "./checkpoints/netflix_multi_100k_30_epochs" \
  --tokenizer "./tokenizers/nm_tokenizer_multi_netflix" \
  --genlen 1000
```

There are a number of generation parameters in this script that can be adjusted. Generation will write by default to `./inference/EXP_1/RUN_1/generated.txt`.

#### 2. Convert raw tokens to PCAP

Convert the raw text generated output to a PCAP using `./generation/conversion.py`.

```bash
uv run ./generation/conversion.py <PATH/TO/GENERATED.TXT> <PATH/TO/OUTPUT/PCAP>
```

**NetSSM does not generate timestamps for packets.** Timestamps are assigned to packets in `conversion.py` by the [Scapy library](https://scapy.net) in the order they are read from `generated.txt` using offsets from the current Unix time. If you would like to sample timestamps from a ground truth PCAP to assign to the generated PCAP, please use the flag `--ts_sample_gt` to specify a path to the desired PCAP to sample from.

## Citation
If you use this codebase, or otherwise find our work valuable, please cite NetSSM:
```tex
PLACEHOLDER
```
