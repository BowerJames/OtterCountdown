# train_grpo.py
import argparse
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from reward_functions import reward_countdown, reward_format
from huggingface_hub import login
from transformers import TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description='Train model using GRPO')
    parser.add_argument('--dataset_name', type=str, default="jimbowyer123/countdown",
                       help='Name of the dataset to use')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-0.5B-Instruct",
                       help='Name or path of the base model')
    parser.add_argument('--output_dir', type=str, default="Qwen2-0.5B-GRPO",
                       help='Directory to save the trained model')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push model to Hub during training')
    parser.add_argument('--hub_model_id', type=str, default="jimbowyer123/OtterCountdown",
                       help='Model ID for uploading to Hub')
    parser.add_argument('--hub_token', type=str, required=True,
                       help='HuggingFace token for pushing to hub')
    return parser.parse_args()

def train(training_args, dataset):   
    if args.hub_token:
        login(token=args.hub_token)
        
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[reward_format, reward_countdown],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split="train")
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=10,
        num_generations=8,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    train(training_args, dataset)
