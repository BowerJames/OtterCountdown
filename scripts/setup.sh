# Create a python environment
python -m venv unsloth-vllm-env && source unsloth-vllm-env/bin/activate

# Update core packages
python -m pip install -U pip setuptools wheel

# Install unsloth
python -m pip install unsloth

# Install fsspec
python -m pip install fsspec[http]==2024.9.0

# Install vllm
git clone -b v0.7.2 https://github.com/vllm-project/vllm.git
cd vllm && python use_existing_torch.py
python -m pip install -r requirements-build.txt && python -m pip install -e . --no-build-isolation
cd ..

# Install Enchant C library
sudo apt install enchant-2

# Install remaining python dependencies
python -m pip install pyenchant wandb

