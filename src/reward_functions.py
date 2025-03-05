import re
import enchant
from reward_utils import countdown_word_completion_score, think_format_score, think_answer_format_score, answer_format_score, hard_format_score, create_clipped_reward
import numpy as np

def reward_think(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the <think> </think> tags.

    A reward will be given for the presence of the <think> </think> tags.

    Input:
        completions: list[list[dict]]
            The completions to reward. The outer list represents each generation, and the inner list represents the messages generated in that generation (the inner list is of length 1). Each dict contains the following keys:
            - "role": str
                The role of the completion.
            - "content": str
                The content of the completion.

    Output:
        rewards: list[float]
            The rewards for each completion.
    """
    # Select the generated text from the completions
    completions = [completion[-1]["content"] for completion in completions]

    # Reward the presence of the <think> tag
    pattern = r"^\n*<think>((.|\n)*)</think>"
    has_think = [bool(re.search(pattern, completion)) for completion in completions]
    rewards = [1.0 if has_think[i] else 0.0 for i in range(len(completions))]
    
    return rewards

def reward_answer(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the <answer> </answer> tags.

    A reward will be given for the presence of the <answer> </answer> tags.

    Input:
        completions: list[list[dict]]
            The completions to reward. The outer list represents each generation, and the inner list represents the messages generated in that generation (the inner list is of length 1). Each dict contains the following keys:
            - "role": str
                The role of the completion.
            - "content": str
                The content of the completion.

    Output:
        rewards: list[float]
            The rewards for each completion.
    """
    # Select the generated text from the completions
    completions = [completion[-1]["content"] for completion in completions]

    # Reward the presence of the <answer> tag
    rewards = [answer_format_score(completion) for completion in completions]

    return rewards

def reward_think_answer(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Reward for the presence of the </think> and <answer> tags.

    A reward will be given for the presence of the </think> and <answer> tags.

    Input:
        completions: list[list[dict]]
            The completions to reward. The outer list represents each generation, and the inner list represents the messages generated in that generation (the inner list is of length 1). Each dict contains the following keys:
            - "role": str
                The role of the completion.
            - "content": str
                The content of the completion.

    Output:
        rewards: list[float]
            The rewards for each completion.
    """
    # Select the generated text from the completions
    completions = [completion[-1]["content"] for completion in completions]

    # Reward the presence of the </think> and <answer> tags 
    rewards = [think_answer_format_score(completion) for completion in completions]

    return rewards

def reward_hard_format(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Hard reward for the presence of the <answer> tag.

    A reward will be given if the completion is formatted perfectly.

    The correct format is <think> thinking </think>\n<answer> answer </answer>.

    Input:
        completions: list[list[dict]]
            The completions to reward. The outer list represents each generation, and the inner list represents the messages generated in that generation (the inner list is of length 1). Each dict contains the following keys:
            - "role": str
                The role of the completion.
            - "content": str
                The content of the completion.

    Output:
        rewards: list[float]
            The rewards for each completion.
    """
    # Select the generated text from the completions
    completions = [completion[-1]["content"] for completion in completions]

    # Check if the completion is formatted correctly
    rewards = [hard_format_score(completion) for completion in completions]
    
    return rewards

def reward_countdown_word(completions: list[dict], reward_data: list[dict], **kwargs) -> list[float]:
    """
    Reward for the countdown letters problem.

    If the reward data indicates it is of task type "countdown_letters", then the reward is the length of the answer if it is a valid english word. the reward is 0 if an invalid word is given or the answer does not fit the letters constraint.

    If the reward data indicates it is not of task type "countdown_letters", then the reward is 0.
    """

    completions = [completion[0]["content"] for completion in completions]

    rewards = [countdown_word_completion_score(completion, data["letters"]) if data["task"] == "countdown_letters" else 0.0 for completion, data in zip(completions, reward_data)]

    return rewards

def reward_global(completions: list[list[dict]], reward_data: list[dict], **kwargs) -> list[float]:
    """
    Global reward function for training the model.
    """
    completions = [completion[-1]["content"] for completion in completions]
    
    rewards = [0.0] * len(completions)



    for i, (completion, data) in enumerate(zip(completions, reward_data)):
        rewards[i] += think_format_score(completion)
        rewards[i] += think_answer_format_score(completion)
        rewards[i] += answer_format_score(completion)
        rewards[i] += hard_format_score(completion)
        rewards[i] += countdown_word_completion_score(completion, data["letters"]) if data["task"] == "countdown_letters" else 0.0

    rewards = create_clipped_reward(rewards)

    return rewards


REWARD_FUNCTIONS = {
    "reward_think": reward_think,
    "reward_answer": reward_answer,
    "reward_think_answer": reward_think_answer,
    "reward_hard_format": reward_hard_format,
    "reward_countdown_word": reward_countdown_word,
    "reward_global": reward_global,
}