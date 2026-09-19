from groq import Groq

# Replace with your actual Groq API key
client = Groq(api_key="gsk_YourActualKeyHere")

try:
    # 1. Fetch available models from Groq
    models = client.models.list()
    print("--- AVAILABLE MODELS ON YOUR ACCOUNT ---")
    for m in models.data:
        print(f"- {m.id}")

    # 2. Test a basic prompt
    print("\n--- TESTING CHAT COMPLETION ---")
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print("Response:", completion.choices[0].message.content)

except Exception as e:
    print("Error:", e)