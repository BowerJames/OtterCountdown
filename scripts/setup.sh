# Create a python environment
python -m venv unsloth-vllm-env && source unsloth-vllm-env/bin/activate

# Update core packages
python -m pip install -U pip setuptools wheel

# Install unsloth
pip install transformers==4.48.3
pip install unsloth

# Install fsspec
pip install fsspec[http]==2024.9.0

# Install triton
git clone https://github.com/triton-lang/triton.git
cd triton/python && pip install ninja cmake
pip install -e . && cd ../../ && rm -rf triton

# Install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm && python use_existing_torch.py
pip install -r requirements-build.txt && pip install -e . --no-build-isolation
cd ..

# Reinstall triton to be compatible with vllm
pip install --force-reinstall triton

# Install Enchant C library
sudo apt install enchant-2

# Install remaining python dependencies
pip install pyenchant wandb

