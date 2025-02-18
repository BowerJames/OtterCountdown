from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
from huggingface_hub import login
from dataclasses import dataclass, asdict

from src.reward_functions import reward_soft_think_open, reward_soft_think_close, reward_soft_answer_open, reward_soft_answer_close, reward_hard_format, reward_countdown_word

@dataclass
class ScriptArgs:
    hub_token: str

@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    test_split: float = 0.01

@dataclass
class BaseModelConfig:
    model_name: str = "meta-llama/meta-Llama-3.1-8B-Instruct"
    max_seq_len: int
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.5

@dataclass
class PeftModelConfig:
    lora_rank: int = 16
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
    lora_alpha: float = 16
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42

def main(base_model_args: BaseModelConfig, peft_model_args: PeftModelConfig, training_args: GRPOConfig, dataset_args: DatasetArgs, script_args: ScriptArgs):
    PatchFastRL("GRPO", FastLanguageModel)

    if script_args.hub_token:
        login(token=script_args.hub_token)

    dataset = load_dataset(dataset_args.dataset_name, split=dataset_args.split)
    dataset = dataset.train_test_split(test_size=dataset_args.test_split)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_args.model_name,
        max_seq_length=base_model_args.max_seq_len,
        load_in_4bit=base_model_args.load_in_4bit,
        fast_inference=base_model_args.fast_inference,
        max_lora_rank=peft_model_args.lora_rank,
        gpu_memory_utilization=base_model_args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_model_args.lora_rank,
        target_modules=peft_model_args.target_modules,
        lora_alpha=peft_model_args.lora_alpha,
        use_gradient_checkpointing=peft_model_args.use_gradient_checkpointing,
        random_state=peft_model_args.random_state,
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        reward_funcs=[reward_soft_think_open, reward_soft_think_close, reward_soft_answer_open, reward_soft_answer_close, reward_hard_format, reward_countdown_word],
        args=training_args,
    )

    trainer.train()
    
    


if __name__ == "__main__":
    parser = TrlParser((BaseModelConfig, PeftModelConfig, GRPOConfig, DatasetArgs, ScriptArgs))
    base_model_args, peft_model_args, training_args, dataset_args, script_args = parser.parse_args_and_config()
    
