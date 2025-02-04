import random
import string
from jinja2 import Template
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

load_dotenv()

prompt_template = """
Given the following letters: {{ letters }}

Make the longest valid english word using the letters. You can only use each letter once. It may not be possible to to use every letter to make a word, in which case just make the longest one possible.

You should think about the problem before you give an answer. Enclose your thoughts within the <think> </think> tags.
After you have finished your thoughts output the best word you have come up with in the format \\boxed{<your_word>}.
""".strip()

def generate_letters():
    # Determine the vowels and consonants
    vowels = "AEIOU"
    consonants = "".join(c for c in string.ascii_uppercase if c not in vowels)

    # Determine the number of consonants and vowels
    num_letters = random.randint(5, 9)
    num_vowels = random.randint(2, num_letters // 2) # Cap max vowels
    num_consonants = num_letters - num_vowels

    # Get random consonants and vowels
    chosen_consonants = random.choices(consonants, k=num_consonants)
    chosen_vowels = random.choices(vowels, k=num_vowels)

    # Combine and shuffle
    letters = chosen_consonants + chosen_vowels
    random.shuffle(letters)

    return " ".join(letters)

def generate_data(num_samples):
    data = {
        "prompt": [],
        "reward_data": []
    }
    template = Template(prompt_template)

    for _ in tqdm(range(num_samples)):
        letters = generate_letters()
        prompt = template.render(letters=letters)
        data["prompt"].append(prompt)
        data["reward_data"].append(
            {
                "task": "countdown_letters",
                "letters": letters,
            }
        )

    dataset = Dataset.from_dict(data)
    
    return dataset

if __name__ == "__main__":
    from huggingface_hub import login
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate countdown letter game data')
    parser.add_argument('-n', '--num-samples', type=int, required=True,
                      help='Number of samples to generate')
    args = parser.parse_args()

    login(os.getenv("HF_TOKEN"))

    dataset = generate_data(args.num_samples)

    dataset.push_to_hub("jimbowyer123/countdown")
    
