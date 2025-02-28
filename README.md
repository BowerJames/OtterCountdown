# OtterCountdown

## Overview

This project is a simple implementation of a countdown solver using a LLM.

## Setup

### Lambda Labs (UNSLOTH)

### Initial Setup

Clone the repository:

```bash
git clone https://github.com/BowerJames/OtterCountdown.git
```

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

Install enchant-2:

```bash
sudo apt install enchant-2
```

Now we can install the remaining dependencies:

```bash
python -m pip install pyenchant wandb
```

### VLLM (Optional)

If you plan to use vllm for generation, you can install it from source with (this step may take 10 - 20 mins):

```bash
git clone -b v0.7.2 https://github.com/vllm-project/vllm.git
```

```bash
cd vllm && python use_existing_torch.py
```

```bash
python -m pip install -r requirements-build.txt && python -m pip install -e . --no-build-isolation
```

### Final Steps

Finally, login to wandb:

```bash
wandb login
```

CD into the repository and launch the training script with your huggingface token:

```bash
cd OtterCountdown && python src/grpo_unsloth.py --config recipe/a100-1x40.yaml --hub-token <your-huggingface-token>
```
