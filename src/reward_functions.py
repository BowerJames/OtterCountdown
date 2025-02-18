import re
import enchant
from src.reward_utils import score_countdown_word_completion

def reward_soft_think_open(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the <think> tag.

    A small reward will be given for the presence of the <think> tag followed by a slight penaltiy for all extra think tags included.

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

    # Initialize the rewards
    rewards = [0.0] * len(completions)

    # Reward the presence of the <think> tag
    presence_reward = 0.25
    extra_tags_penalty = presence_reward / 2
    pattern = r"<think>"
    num_tags = [len(re.findall(pattern, completion)) for completion in completions]
    rewards = [rewards[i] + presence_reward - (extra_tags_penalty * (num_tags[i] - 1)) if num_tags[i] > 0 else rewards[i] for i in range(len(completions))]

    return rewards

def reward_soft_think_close(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the </think> tag.

    A small reward will be given for the presence of the </think> tag followed by a slight penaltiy for all extra think tags included.

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

    # Initialize the rewards
    rewards = [0.0] * len(completions)

    # Reward the presence of the </think> tag
    presence_reward = 0.25
    extra_tags_penalty = presence_reward / 2
    pattern = r"</think>"
    num_tags = [len(re.findall(pattern, completion)) for completion in completions]
    rewards = [rewards[i] + presence_reward - (extra_tags_penalty * (num_tags[i] - 1)) if num_tags[i] > 0 else rewards[i] for i in range(len(completions))]

    return rewards

def reward_soft_answer_open(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the <answer> tag.

    A small reward will be given for the presence of the <answer> tag followed by a slight penaltiy for all extra answer tags included.

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

    # Initialize the rewards
    rewards = [0.0] * len(completions)

    # Reward the presence of the <answer> tag
    presence_reward = 0.25
    extra_tags_penalty = presence_reward / 2
    pattern = r"<answer>"
    num_tags = [len(re.findall(pattern, completion)) for completion in completions]
    rewards = [rewards[i] + presence_reward - (extra_tags_penalty * (num_tags[i] - 1)) if num_tags[i] > 0 else rewards[i] for i in range(len(completions))]

    return rewards

def reward_soft_answer_close(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Soft reward for the presence of the </answer> tag.

    A small reward will be given for the presence of the </answer> tag followed by a slight penaltiy for all extra answer tags included.

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

    # Initialize the rewards
    rewards = [0.0] * len(completions)

    # Reward the presence of the </answer> tag
    presence_reward = 0.25
    extra_tags_penalty = presence_reward / 2
    pattern = r"</answer>"
    num_tags = [len(re.findall(pattern, completion)) for completion in completions]
    rewards = [rewards[i] + presence_reward - (extra_tags_penalty * (num_tags[i] - 1)) if num_tags[i] > 0 else rewards[i] for i in range(len(completions))]

    return rewards

def reward_hard_format(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Hard reward for the presence of the <answer> tag.

    A reward will be given if the completion is formatted perfectly.

    The correct format is <think>\nquestion\n</think>\n<answer>\nanswer\n</answer>.

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

    # Initialize the rewards
    rewards = [0.0] * len(completions)
    
    # Check if the completion is formatted correctly
    pattern = r"^<think>\n(.*?)\n</think>\n<answer>\n(.*?)\n</answer>\n$"
    num_matches = [len(re.findall(pattern, completion)) for completion in completions]
    rewards = [rewards[i] + 1.0 if num_matches[i] > 0 else rewards[i] for i in range(len(completions))]
    
    return rewards

def reward_countdown_word(completions: list[dict], reward_data: list[dict], **kwargs) -> list[float]:
    """
    Reward for the countdown letters problem.

    If the reward data indicates it is of task type "countdown_letters", then the reward is the length of the answer if it is a valid english word. the reward is 0 if an invalid word is given or the answer does not fit the letters constraint.

    If the reward data indicates it is not of task type "countdown_letters", then the reward is 0.
    """

    completions = [completion[0]["content"] for completion in completions]

    rewards = [score_countdown_word_completion(completion, data["letters"]) if data["task"] == "countdown_letters" else 0.0 for completion, data in zip(completions, reward_data)]

    return rewards
