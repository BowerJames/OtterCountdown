# train_grpo.py
import argparse
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig, get_peft_config, LogCompletionsCallback
from reward_functions import reward_countdown, reward_format
from huggingface_hub import login
from transformers import TrainingArguments, AutoModelForCausalLM
from dataclasses import dataclass

@dataclass
class ScriptArgs:
    dataset_name: str
    hub_token: str



def train(script_args: ScriptArgs, model_args: ModelConfig, training_args: TrainingArguments):  

    if script_args.hub_token:
        login(token=script_args.hub_token)

    dataset = load_dataset(script_args.dataset_name, split="train")
    dataset = dataset.rename_column("messages", "prompts")
    dataset = dataset.train_test_split(test_size=64)

    peft_config = get_peft_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, load_in_8bit=model_args.load_in_8bit, trust_remote_code=True)
        
    trainer = GRPOTrainer(
        model=model_args.model_name,
        reward_funcs=[reward_format, reward_countdown],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config
    )
    completions_callback = LogCompletionsCallback(
        trainer=trainer,
        num_prompts=8
    )
    trainer.add_callback(completions_callback)
    trainer.train()

if __name__ == "__main__":
    parser = TrlParser((ScriptArgs, ModelConfig, GRPOConfig))
    script_args, model_args, training_args = parser.parse_args_and_config()

    train(script_args, model_args, training_args)
