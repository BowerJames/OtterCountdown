import re
import enchant

def reward_format(completions, **kwargs) -> list[float]:

    completions = [completion[0]["content"] for completion in completions]

    # Search for the <think> </think> tags
    pattern = r"<think>(.*?)</think>"
    has_thoughts = [len(re.findall(pattern, completion)) > 0 for completion in completions]

    # Search for the \\boxed{<your_word>} tags
    pattern = r"\\boxed{(.*?)}"
    has_word = [len(re.findall(pattern, completion)) > 0 for completion in completions]

    # Score is 1.0 if both are true, 0.0 otherwise
    return [int(has_thoughts[i] and has_word[i]) for i in range(len(completions))]

def reward_countdown(completions, reward_data, **kwargs):

    completions = [completion[0]["content"] for completion in completions]

    # Get the letters
    letters = [data["letters"].split(" ") for data in reward_data]
    # Extract the answer from the completion
    pattern = r"\\boxed{(.*?)}"
    answers = [re.findall(pattern, completion) for completion in completions]
    answers = [answer[0].strip() if len(answer) > 0 else "" for answer in answers]

    # Check if the answer is a valid english word
    d = enchant.Dict("en_GB")
    answers = [answer if len(answer) > 0 and d.check(answer) else "" for answer in answers]
    # Check if the answer only uses available letters
    scores = []
    for answer, letters in zip(answers, letters):
        if answer == "":
            scores.append(0.0)
            continue
            
        # Convert answer and letters to uppercase for comparison
        answer = answer.upper()
        letters_available = letters.copy()
        
        # Check each letter in answer
        valid = True
        for letter in answer:
            if letter not in letters_available:
                valid = False
                break
            letters_available.remove(letter)
            
        # Score based on length if valid, 0 otherwise
        scores.append(float(len(answer)) if valid else 0.0)

    return scores

