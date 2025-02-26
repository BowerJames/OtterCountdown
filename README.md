# OtterCountdown

## Overview

This project is a simple implementation of a countdown solver using a LLM.

## Setup

### Lambda Labs (UNSLOTH)

Create a python environment:

```bash
python -m venv unsloth-vllm-env && source unsloth-vllm-env/bin/activate
```

Make sure that pip, setuptools, and wheel are all up to date:

```bash
python -m pip install -U pip setuptools wheel
```

Install unsloth:

```bash
python -m pip install unsloth
```

Install fsspec compatible with huggign face datasets library:

```bash
python -m pip install fsspec[http]==2024.9.0
```

Now build vllm:

```bash
git clone -b v0.7.2 https://github.com/vllm-project/vllm.git
```

```bash
cd vllm && python use_existing_torch.py
```

```bash
python -m pip install -r requirements-build.txt && python -m pip install -e . --no-build-isolation
```

This step may take 10 - 20 mins.

Before we can install the remaining dependencies, we need to install the Enchant C library:

```bash
sudo apt install enchant-2
```

Now we can install the remaining dependencies:

```bash
python -m pip install pyenchant wandb
```

Finally, login to wandb:

```bash
wandb login
```

Now launch the training script:

```bash
python src/grpo_unsloth.py --config recipe/a100-1x40.yaml --hub-token <your-huggingface-token>
```
