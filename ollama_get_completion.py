import time
import ollama

def get_ollama_completion(prompt, 
                         model_name="phi4", 
                         max_tokens=500, 
                         temperature=0.7, 
                         top_p=0.9):
    """
    Streams text completion from an Ollama model and measures tokens per second (TPS).

    :param prompt: The input text prompt.
    :param model_name: The Ollama model name or alias (e.g. 'llama3.1').
    :param max_tokens: Maximum number of tokens to generate.
    :param temperature: The temperature for randomness.
    :param top_p: The nucleus sampling hyperparameter for decoding.
    :return: Estimated tokens per second (TPS).
    """

    # Start timing
    start_time = time.time()

    # Track tokens
    token_count = 0
    generated_text = ""

    # Ollama streaming API call
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            stream=True  # Enable streaming
        )

        # Process streamed output
        for chunk in response:
            token = chunk["message"]["content"]  # Extract text from chunk
            print(token, end="", flush=True)  # Stream output to console
            generated_text += token
            token_count += len(token.strip().split())  # Estimate token count

    except Exception as e:
        print(f"\nError during generation: {e}")
        return None

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

    # Print stats
    print(f"\n\nTokens Generated: {token_count}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Tokens Per Second (TPS): {tokens_per_second:.2f}\n")

    return tokens_per_second

if __name__ == "__main__":
    prompt_text = "Roger has 5 apples. He buys 3 more, then gives away 2. How many does he have now?"
    # prompt_text="""Translate the following into French: 
    #           'Hello' → 'Bonjour', 
    #           'Goodbye' → 'Au revoir', 
    #           'Thank you' → ?"""

    # prompt_text = """In quantum physics, a “Quaron” is a hypothetical particle that exists in two places at once. Here’s how it’s used in a sentence:
    #                 “Scientists theorize that a Quaron could explain some of the mysteries of teleportation.”
    #                 Now, use “Neuroflux” in a sentence."""
    
    # prompt_text = """1️ Generate a product name →
    #                  2️ Generate a tagline →
    #                  3️ Write a product description"""
    

    # prompt_text = """Classify the sentiment of this review as either 'Positive' or 'Negative'. 
    #           Text: 'The food was cold and tasteless.' 
    #           Sentiment:" """
    # prompt_text = """Please translate 'Thank you' into Telugu language.Do not ouput other than translation"""

    prompt_text = "How are you doing today?"
    tps = get_ollama_completion(prompt_text, 
                               model_name="phi4",  # Use your local model alias
                               max_tokens=50, 
                               temperature=0)

    if tps is not None:
        print(f"Approx. Tokens per second: {tps:.2f}")