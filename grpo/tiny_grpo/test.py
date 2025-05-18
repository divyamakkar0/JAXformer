from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from gsm8k_data import SYSTEM_PROMPT, build_prompt, extract_answer_from_model_output


def main():
    """
    Main function to load the fine-tuned model and test it on example math problems.

    Explanation:
        1. Determines the device (GPU if available, otherwise CPU).
        2. Loads the fine-tuned model and tokenizer from the saved path.
        3. Tests the model on predefined math problems.
        4. Formats the prompt using the same SYSTEM_PROMPT and build_prompt function as training.
        5. Generates and displays responses for each test prompt.
    """
    # Determine the device: use GPU if available, else fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the saved model and tokenizer
    saved_model_path = "grpo_finetuned_model"

    # Load the model
    loaded_model = AutoModelForCausalLM.from_pretrained(
        saved_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    loaded_tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    # Define test prompts
    prompts_to_test = [
        "How much is 1+1?",
        "I have 3 apples, my friend eats one and I give 2 to my sister, how many apples do I have now?",
        "Solve the equation 6x + 4 = 40",
    ]

    # Test each prompt
    for prompt in prompts_to_test:
        # Prepare the prompt using the same format as during training
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        test_prompt = build_prompt(test_messages)

        # Tokenize the prompt and generate a response
        test_input_ids = loaded_tokenizer.encode(test_prompt, return_tensors="pt").to(
            device
        )

        # Generate response with similar parameters to those used in training
        with torch.no_grad():
            test_output_ids = loaded_model.generate(
                test_input_ids,
                max_new_tokens=400,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=loaded_tokenizer.pad_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id,
                do_sample=True,
                early_stopping=False,
            )

        test_response = loaded_tokenizer.decode(
            test_output_ids[0], skip_special_tokens=True
        )

        # Print the test prompt and the model's response
        print("\nTest Prompt:")
        print(test_prompt)
        print("\nModel Response:")
        print(test_response)

        # Extract and display the answer part for easier evaluation
        try:
            extracted_answer = extract_answer_from_model_output(test_response)
            print("\nExtracted Answer:")
            print(extracted_answer)
            print("-" * 50)
        except Exception as e:
            print(f"\nFailed to extract answer: {e}")
            print("-" * 50)


if __name__ == "__main__":
    main()
