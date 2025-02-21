import enchant
import re

def is_valid_word(word: str) -> bool:
    """Check if a word is a valid english word."""
    d = enchant.Dict("en_GB")
    return d.check(word)

def validate_letters_constraint(word: str, letters: str) -> bool:
    """Check if a word uses the letters constraint."""
    valid = True
    letters = letters.split(" ")
    for letter in word:
        if letter not in letters:
            valid = False
            break
        letters.remove(letter)
    return valid

def extract_answer(completion: str) -> str:
    """Extract the answer from a completion."""

    pattern = r"<answer>((.|\n)*)</answer>"
    match = re.search(pattern, completion)
    if match:
        return match.group(1).strip().upper()
    else:
        return ""
    
def score_countdown_word_completion(completion: str, letters: str) -> float:
    """Score a countdown word completion."""
    answer = extract_answer(completion)
    if not answer:
        return 0.0
    if not is_valid_word(answer):
        return 0.0
    if not validate_letters_constraint(answer, letters):
        return 0.0
    return float(len(answer))
