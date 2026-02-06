import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load the .env file
load_dotenv()

# 2. DEBUG: Print the key (first 5 chars only for safety)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found. Check your .env file location.")
else:
    print(f"✅ Key loaded successfully: {api_key[:5]}...")

# 3. Initialize Client
try:
    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello Groq!"}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    print("--- Response ---")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print(f"❌ Connection failed: {e}")