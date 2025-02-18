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
pip install unsloth
```

Install fsspec compatible with huggign face datasets library:

```bash
pip install fsspec[http]==2024.9.0
```

Now build vllm from source. Start by build triton:

```bash
git clone https://github.com/triton-lang/triton.git
```

```bash
cd triton/python && pip install ninja cmake
```

```bash
pip install -e . && cd ../../ && rm -rf triton
```

Now build vllm:

```bash
git clone https://github.com/vllm-project/vllm.git
```

```bash
cd vllm && python use_existing_torch.py
```

```bash
pip install -r requirements-build.txt && pip install -e . --no-build-isolation
```

The final step may take 10 - 20 mins.

Now we need to force reinstall triton:

```bash
pip install --force-reinstall triton
```

Before we can install the remaining dependencies, we need to install the Enchant C library:

```bash
sudo apt install enchant-2
```

Now we can install the remaining dependencies:

```bash
pip install pyenchant wandb
```

Finally, login to wandb:

```bash
wandb login
```

Now you are good to go.