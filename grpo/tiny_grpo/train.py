from transformers import AutoModelForCausalLM, AutoTokenizer
import random 
import wandb 
import torch 
import os 
from contextlib import nullcontext
from gsm8k_data import prepare_dataset, reward_fn, extract_gsm8k_answer, gsm8k_metric
from grpo import train_with_grpo, optimize_model_memory


def setup_training_environment(device_name, dtype = "bfloat16"):
    """set up mixed precision (for memory optimization)"""
    
    # random seed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # mixed precision
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return {
        'device': device_name,
        'ctx': ctx,
        'device_type': device_type,
    }
    
env = setup_training_environment("cuda" if torch.cuda.is_available() else "cpu")
    
def evaluate_model(model, tokenizer, eval_examples, device, env, max_completion_length):
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # tokenisze & generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           with env['ctx']:
               outputs = model.generate(
                   inputs,
                   max_new_tokens=max_completion_length,
                   temperature=0.7,
                   num_return_sequences=1,
                   pad_token_id=tokenizer.pad_token_id,
                   eos_token_id=tokenizer.eos_token_id,
                   forced_eos_token_id=tokenizer.eos_token_id,
                   early_stopping=False,
                ) # 'forward generation --> RL on reward'
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       try:           
           # 1. extract functional 
           predicted = extract_gsm8k_answer(response)
           # 2. metric-based reward fn
           is_correct, _ = gsm8k_metric(predicted, expected)

           # update counter for correct answers
           if is_correct:
               correct += 1

           # printing the metrics
           print("\nPrompt:")
           print(full_prompt)
           print("\nExpected Answer:")
           print(expected)
           print("\nExtracted Answer:")
           print(predicted)
           print("\nFull Generated Response:")
           print(response)
           print("\nCorrect:", "✓" if is_correct else "✗")
           print("-"*50)

       except Exception as e:
           print("\nFailed to parse model output for prompt:")
           print(full_prompt)
           print("Error:", e)
           print("-"*50)

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   print("="*50)

   # Return model to training mode
   model.train()
   return accuracy

def main(args): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print(f"Using primary device: {device}")

    model_name = args.model_name
    output_dir = args.output_dir

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model downloaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    size_of_eval_data = 30 # change to a smaller value to save time or to a larger number for a more reliable estimate
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]

    model = optimize_model_memory(model)
    
    print("\nInitial model evaluation before finetuning:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device, env, args.max_completion_length)
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
    print("Weights & Biases initialized.")

    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=reward_fn,
        num_iterations=args.num_iterations,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.learning_rate,
        mu=args.mu,
        epsilon=args.epsilon,
        env=env
    )

    wandb.finish() 
    print("Training completed and wandb run finished.")

    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device, env, args.max_completion_length)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    print("\nSaving GRPO fine-tuned model...")
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with GRPO on GSM8K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save the model")
    parser.add_argument("--num_iterations", type=int, default=2, 
                        help="Number of iterations")
    parser.add_argument("--num_steps", type=int, default=400, 
                        help="Number of steps per iteration")
    parser.add_argument("--batch_size", type=int, default=6, 
                        help="Batch size")
    parser.add_argument("--num_generations", type=int, default=8, 
                        help="Number of generations per example")
    parser.add_argument("--max_completion_length", type=int, default=512, 
                        help="Maximum completion length")
    parser.add_argument("--beta", type=float, default=0.1, 
                        help="Beta parameter for GRPO")
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                        help="Learning rate")
    parser.add_argument("--mu", type=float, default=6, 
                        help="Mu parameter for GRPO")
    parser.add_argument("--epsilon", type=float, default=0.1, 
                        help="Epsilon parameter for GRPO")
    
    args = parser.parse_args()
    main(args)