import enchant
import re
import numpy as np

def is_valid_word(word: str) -> bool:
    """Check if a word is a valid english word."""
    d = enchant.Dict("en_GB")
    return d.check(word)

def validate_letters_constraint(word: str, letters: str) -> bool:
    """Check if a word uses the letters constraint."""
    valid = True
    word = word.upper()
    letters = letters.upper()
    letters = letters.split(" ")
    for letter in word:
        if letter not in letters:
            valid = False
            break
        letters.remove(letter)
    return valid

def extract_answer(completion: str) -> str:
    """Extract the answer from a completion."""

    pattern = r"<answer>(.|\n)*</answer>"
    match = re.search(pattern, completion)
    if match:
        return match.group(1).strip().upper()
    else:
        return ""
    
def countdown_word_completion_score(completion: str, letters: str) -> float:
    """Score a countdown word completion."""
    answer = extract_answer(completion)
    if not answer:
        return 0.0
    elif not is_valid_word(answer):
        return 0.0
    elif not validate_letters_constraint(answer, letters):
        return 0.0
    else:
        return float(len(answer)) / 9

def think_format_score(completion: str) -> float:
    """Score for the presence of the <think> tag."""

    pattern = r"^\n*<think>(.|\n)*</think>"
    has_think = bool(re.search(pattern, completion))
    return 0.25 if has_think else 0.0
    
def think_answer_format_score(completion: str) -> float:
    """Reward for the presence of the </think> and <answer> tags."""
    pattern = r"</think>(.|\n)*<answer>"
    has_think_answer = bool(re.search(pattern, completion))
    return 0.25 if has_think_answer else 0.0

def answer_format_score(completion: str) -> float:
    """Reward for the presence of the <answer> tag."""
    pattern = r"<answer>(.|\n)*</answer>\n*$"
    has_answer = bool(re.search(pattern, completion))
    return 0.25 if has_answer else 0.0

def hard_format_score(completion: str) -> float:
    """Reward for the presence of the <think> and <answer> tags."""
    pattern = r"^\n*<think>(.|\n)*</think>\n*<answer>(.|\n)*</answer>\n*$"
    has_think_answer = bool(re.search(pattern, completion))
    return 0.25 if has_think_answer else 0.0

def create_clipped_reward(rewards: list[float]) -> list[float]:
    """Function that takes a list of floats and returns a list of floats such that the mean is the same but all values above the mean have the same value, all values below the mean have the same value and values equal to the mean have the value of the mean."""
    if not rewards:
        return []
        
    rewards_np = np.array(rewards)
    mean = np.mean(rewards_np)
    epsilon = 1e-7  # Small epsilon for numerical precision
    
    # Count values above, below and equal to mean
    above_mask = rewards_np > mean + epsilon
    below_mask = rewards_np < mean - epsilon
    equal_mask = np.abs(rewards_np - mean) <= epsilon
    
    n_above = np.sum(above_mask)
    n_below = np.sum(below_mask)
    n_equal = np.sum(equal_mask)
    
    # Calculate values that preserve the mean
    if n_above > 0:
        above_value = mean + (1 / n_above)
    else:
        above_value = mean
        
    if n_below > 0:
        below_value = mean - (1 / n_below)
    else:
        below_value = mean
        
    # Create clipped rewards using masks
    clipped = np.where(above_mask, above_value,
                      np.where(below_mask, below_value, mean))
    
    return clipped.tolist()


