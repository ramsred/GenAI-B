import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
# === Chat Completion ===
chat_response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather like today in Raleigh?"}
    ]
)

print("\n💬 Chat Response:")
print(chat_response.choices[0].message.content)