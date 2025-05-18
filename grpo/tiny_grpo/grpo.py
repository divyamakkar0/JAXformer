import copy
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
from contextlib import nullcontext


def setup_training_environment(device_name, dtype="bfloat16"):
    """set up mixed precision (for memory optimization)"""

    # Set up random seed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up mixed precision
    device_type = "cuda" if "cuda" in device_name else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    return {
        "device": device_name,
        "ctx": ctx,
        "device_type": device_type,
    }


def get_memory_usage():
    """Get current GPU memory usage in a human-readable format."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CUDA not available"


def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility across Python, NumPy, and PyTorch."""
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selective_log_softmax(logits, input_ids, chunk_size=64):
    """Process in chunks to reduce peak memory"""
    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape
    log_probs = torch.zeros(batch_size, seq_len, device=device)

    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk_logits = logits[:, i:end_idx, :]
        chunk_ids = input_ids[:, i:end_idx]
        chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
        # print(" - chunkwise softmax computation GPU memory: ", get_memory_usage())
        log_probs[:, i:end_idx] = chunk_log_probs.gather(
            dim=-1, index=chunk_ids.unsqueeze(-1)
        ).squeeze(-1)
        del chunk_logits, chunk_log_probs
        torch.cuda.empty_cache()
    return log_probs


def compute_log_probs(
    model, input_ids, attention_mask, logits_to_keep, env, chunk_size=64
):
    with env["ctx"]:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            :, :-1, :
        ]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids, chunk_size)


def create_completion_mask(completion_ids, eos_token_id):

    # ----- TBD: replace this with a less hacky solution -----
    is_eos = completion_ids == eos_token_id
    # shape: (batch_size,) value: max length of completion
    eos_idx = torch.full(
        (is_eos.size(0),),
        is_eos.size(1),
        dtype=torch.long,
        device=completion_ids.device,
    )
    mask_exists = is_eos.any(dim=1)  # operate on sequence with eos_token
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[
        mask_exists
    ]  # first eos token index ｜ hacky: argmax returns first index of largest value
    # ----- TBD: replace above with a less hacky solution -----

    # create indices of tokens, then build non-end mask by comparing with eos_token index (first appearance indicate termination)
    sequence_indices = torch.arange(
        is_eos.size(1), device=completion_ids.device
    ).expand(
        is_eos.size(0), -1
    )  # (batch_size, max_length)
    return (sequence_indices < eos_idx.unsqueeze(1)).int()


def generate_completions(
    model,
    tokenizer,
    prompts,
    device,
    num_generations=4,
    max_completion_length=32,
    env=None,
):
    """Generate multiple completions for each prompt, record completion mask (end-of-sequence)"""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(
        1
    )  # Question: sequences within batch should have different length? This leads to wrong completion mask?
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    with env["ctx"]:
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False,
        )  # Important: same length outputs from this generate function
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(
        completion_ids, tokenizer.eos_token_id
    )  # completion mask excludes eos_token and suffix tokens
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def generate_rollout_data(
    model,
    ref_model,
    tokenizer,
    batch_samples,
    device,
    num_generations,
    max_completion_length,
    env,
    chunk=64,
):
    """Generate responses and calculate log-probabilities of each response under two model"""
    prompts = [
        sample["prompt"] if isinstance(sample, dict) else sample[0]
        for sample in batch_samples
    ]
    answers = [
        sample["answer"] if isinstance(sample, dict) else sample[1]
        for sample in batch_samples
    ]
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model,
            tokenizer,
            prompts,
            device,
            num_generations,
            max_completion_length,
            env,
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # same question: whether dim=1 is same across in-batch sequences
        # AR perplexity based RL (Issue #1) --> can we use 'skippy loss'
        # hidden-space RL (Idea #1) --> rollout on latent space?

        old_log_probs = compute_log_probs(
            model, input_ids, attention_mask, logits_to_keep, env, chunk
        )
        ref_log_probs = compute_log_probs(
            ref_model, input_ids, attention_mask, logits_to_keep, env, chunk
        )  # ref is base model ? (also used in KL regularization I recall)

    # one-turn completion & reward assignment (Issue #2)
    formatted_completions = [
        [{"content": tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }


# Issue #4. do we need to keep all 3 model? prev, curr, base?
def grpo_loss(
    model,
    ref_model,
    rollout_data,
    tokenizer,
    reward_function,
    device,
    beta=0.01,
    epsilon=0.2,
    env=None,
    chunk=64,
):
    """
    GRPO loss function:
    - group normalized reward
    - conservative advantage clipping
    - kl regularization
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]

    new_log_probs = compute_log_probs(
        model, input_ids, attention_mask, logits_to_keep, env, chunk
    )

    # ratio: between new and old
    ratio = torch.exp(new_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(
            completions=rollout_data["formatted_completions"],
            answers=rollout_data["repeated_answers"],
        ),
        dtype=torch.float32,
        device=device,
    )

    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]

    # group refers to 'num_genereations' over each prompt, literally 'best-of-N', relative advantage is calculated here
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surrogate_loss = torch.min(
        ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    )
    kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -(
        (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    ).mean()
    del kl, surrogate_loss, per_token_loss

    return loss, avg_reward


def train_with_grpo(
    model,
    tokenizer,
    train_data,
    num_iterations=1,
    num_steps=500,
    batch_size=4,
    num_generations=4,
    max_completion_length=128,
    beta=0.1,
    learning_rate=5e-6,
    mu=3,
    epsilon=0.2,
    reward_function=None,
    env=None,
    gradient_accumulation_steps=1,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created")

        # re-initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        for step in range(num_steps):
            print(f"\nStep {step+1}/{num_steps}")
            batch_samples = random.sample(train_data, batch_size)

            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    device,
                    num_generations,
                    max_completion_length,
                    env,
                )
                print(
                    "\n\n------------------------ \n Example response: \n",
                    rollout_data["formatted_completions"][0][0]["content"],
                )
                # Clear cache after generating rollouts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for grpo_iter in range(mu):
                print(f"GRPO inner loop {grpo_iter+1}/{mu}")
                loss, avg_reward = grpo_loss(
                    model,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    device=device,
                    beta=beta,
                    epsilon=epsilon,
                    env=env,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # Clear cache after each iteration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # log to wandb
                wandb.log(
                    {
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                    }
                )

                print(
                    f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                    f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}, reward: {avg_reward}"
                )

                del loss  # explicitly delete loss to free memory

    return model


def optimize_model_memory(model):
    model.config.use_cache = False
    model = torch.compile(model)
    model.gradient_checkpointing_enable()
    return model
