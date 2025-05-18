# util for data preparation (GSM8K) dataset

import re
from datasets import load_dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_gsm8k_answer(text):
    """
    Extracts the value from the last <answer>...</answer> tag in the text.
    - very strict without much diversity in the extractor functional here
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_gsm8k_answer_from_dataset(text):
    """
    Extracts the answer from the GSM8K dataset examples.
    - specific prompt ask for direct answer following '####' symbol
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def build_prompt(messages):
    """simple change line combination of all response without any identifier whatsoever"""
    return "\n".join([msg["content"].strip() for msg in messages])


def prepare_dataset(split="train"):
    """Load and prepare the GSM8K dataset for training with string prompts."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []
    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_gsm8k_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)
    return formatted_data


def extract_last_number(text):
    text = text.replace("$", "").replace("%", "")
    pattern = r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def extract_single_number(text):
    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None


def gsm8k_metric(predicted: str, expected: str) -> tuple[bool, float]:
    if predicted == expected:  # Exact match
        is_correct = True
        reward = 2.0
    else:
        # Try single number matching
        pred_num = extract_single_number(str(predicted))
        exp_num = extract_single_number(str(expected))
        if pred_num is not None and exp_num is not None and pred_num == exp_num:
            is_correct = True
            reward = 1.5
        else:
            # the way I view this, it's just a metric-based evaluation functional
            # Try last number matching
            pred_num = extract_last_number(str(predicted))
            exp_num = extract_last_number(str(expected))
            is_correct = (
                pred_num is not None and exp_num is not None and pred_num == exp_num
            )
            reward = 0.0
    return is_correct, reward


def functional_reward_fn(completions, answers):
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_gsm8k_answer(response) for response in responses]
    rewards = []
    for pred, exp in zip(extracted, answers):
        is_correct, reward = gsm8k_metric(pred, exp)
        rewards.append(reward)
    # count number of words in the response (not token, not character, words) (interesting...)
    completion_lengths = [len(response.split()) for response in responses]
    return rewards


def structural_reward_fn(completions):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    format_scores = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
        format_scores.append(score)
    return rewards


# trial and error here, for better weightage between the reward components ...
def reward_fn(completions, answers):
    functional_rewards = functional_reward_fn(completions, answers)
    structural_rewards = structural_reward_fn(completions)

    combined_rewards = []
    for f_score, s_score in zip(functional_rewards, structural_rewards):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(f_score + s_score)

    return combined_rewards
