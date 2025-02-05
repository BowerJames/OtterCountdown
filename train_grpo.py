# train_grpo.py
import argparse
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
from reward_functions import reward_countdown, reward_format
from huggingface_hub import login
from transformers import TrainingArguments
import wandb
from dataclasses import dataclass

@dataclass
class ScriptArgs:
    dataset_name: str
    hub_token: str

    

def train(script_args, model_args, training_args):  

    if script_args.hub_token:
        login(token=script_args.hub_token)

    dataset = load_dataset(script_args.dataset_name, split="train")
        
    trainer = GRPOTrainer(
        model=model_args.model_name,
        reward_funcs=[reward_format, reward_countdown],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    parser = TrlParser((ScriptArgs, ModelConfig, GRPOConfig))
    script_args, model_args, training_args = parser.parse_args_and_config()

    train(script_args, model_args, training_args)
