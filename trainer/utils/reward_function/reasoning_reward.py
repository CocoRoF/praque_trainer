## This example is inspired by the format reward function used in the paper
## DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
## It is designed for conversational format, where prompts and completions consist of structured messages.

import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

## All columns names (but prompt) that the dataset may have can be included in kwargs (like ground_truth).
## For example, if the dataset contains a column named ground_truth, the function will be called with ground_truth as a keyword argument.

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
