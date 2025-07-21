import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading

def get_llama_completion(prompt, model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=500, temperature=0.7):
    """
    Streams text completion from Meta's LLaMA 3.2 3B model and measures tokens per second.

    :param prompt: The input text prompt.
    :param model_name: The LLaMA model to use.
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: The temperature for randomness.
    :return: Tokens per second (TPS).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Set pad token ID to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Run generation in a separate thread and start timing
    start_time = time.time()
    
    thread = threading.Thread(target=model.generate, kwargs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    })
    thread.start()

    # Count generated tokens
    token_count = 0
    for token in streamer:
        print(token, end="", flush=True)
        token_count += 1

    thread.join()  # Ensure generation completes
    end_time = time.time()

    # Calculate tokens per second
    elapsed_time = end_time - start_time
    tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\nTokens Generated: {token_count}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Tokens Per Second (TPS): {tokens_per_second:.2f}")

    return tokens_per_second

# Example Usage
if __name__ == "__main__":
    # prompt_text = "How are you doing today?"
    prompt_text = "Roger has 5 apples. He buys 3 more, then gives away 2. How many does he have? Let's think step by step.And do not repeat the answer or question"
    # prompt_text = "Give me a long story about family and friendship in 100 words."
    tps = get_llama_completion(prompt_text)
    print(f"Tokens per second: {tps:.2f}")


