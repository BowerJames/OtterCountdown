from unsloth import FastLanguageModel, PatchFastRL
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
from huggingface_hub import login
from dataclasses import dataclass, asdict

from reward_functions import reward_think, reward_answer, reward_think_answer, reward_hard_format, reward_countdown_word

@dataclass
class DatasetArgs:
    dataset_name: str
    split: str = "train"
    system_prompt: str | None = "You are a reasoning agent. You will be given a problem to solve.\n\nYou should think about the problem before you give an answer.\n\nEnclose your thoughts within the <think> </think> tags and provide your answer within the <answer> </answer> tags.\n\nThe complete format of your response should be:\n\n<think>\n...\n</think>\n<answer>\n...\n</answer>"

@dataclass
class BaseModelConfig:
    model_name: str = "meta-llama/meta-Llama-3.1-8B-Instruct"
    max_seq_len: int = 2048
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.5

@dataclass
class PeftModelConfig:
    target_modules: list[str]
    lora_rank: int = 16
    lora_alpha: float = 16
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42
    use_dora: bool = False

def get_dataset(dataset_args: DatasetArgs) -> Dataset:
    dataset = load_dataset(dataset_args.dataset_name, split=dataset_args.split)

    def map_to_conversation(example):
        prompt = example["prompt"]
        if isinstance(prompt, str):
            prompt = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        if dataset_args.system_prompt:
            prompt.insert(0, {
                "role": "system",
                "content": dataset_args.system_prompt
            })

        example["prompt"] = prompt
        return example
    
    dataset = dataset.map(map_to_conversation)
    return dataset

def main(base_model_args: BaseModelConfig, peft_model_args: PeftModelConfig, training_args: GRPOConfig, dataset_args: DatasetArgs):
    PatchFastRL("GRPO", FastLanguageModel)

    if training_args.hub_token:
        login(token=training_args.hub_token)

    dataset = get_dataset(dataset_args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_args.model_name,
        max_seq_length=base_model_args.max_seq_len,
        load_in_4bit=base_model_args.load_in_4bit,
        fast_inference=base_model_args.fast_inference,
        max_lora_rank=peft_model_args.lora_rank,
        gpu_memory_utilization=base_model_args.gpu_memory_utilization,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_model_args.lora_rank,
        target_modules=peft_model_args.target_modules,
        lora_alpha=peft_model_args.lora_alpha,
        use_gradient_checkpointing=peft_model_args.use_gradient_checkpointing,
        random_state=peft_model_args.random_state,
        use_dora=peft_model_args.use_dora,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_think, reward_answer, reward_think_answer, reward_hard_format, reward_countdown_word],
        args=training_args,
    )

    trainer.train()
    
    


if __name__ == "__main__":
    parser = TrlParser((BaseModelConfig, PeftModelConfig, GRPOConfig, DatasetArgs))
    base_model_args, peft_model_args, training_args, dataset_args = parser.parse_args_and_config()

    main(base_model_args, peft_model_args, training_args, dataset_args)
    
