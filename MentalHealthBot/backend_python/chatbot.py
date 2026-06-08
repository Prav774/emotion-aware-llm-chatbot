import os
from groq import Groq
from collections import deque

from predict import predict  # your LSTM function

# load API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔥 memory (last 3 messages)
memory = deque(maxlen=3)


def generate_response(user_input):
    # step 1: sentiment
    sentiment = predict(user_input)

    # step 2: store memory
    memory.append(user_input)

    context = "\n".join(memory)

    # step 3: build prompt
    prompt = f"""
You are a mental health support assistant.

User sentiment: {sentiment}

Conversation history:
{context}

User says: {user_input}

Respond in a supportive, human, empathetic way.
If negative → give coping suggestions.
If positive → encourage and engage.
Keep response natural (2-3 lines max).
"""

    # step 4: call Groq LLM
    chat = client.chat.completions.create(
        model="llama3-70b-8192",   # strong model
        messages=[
            {"role": "system", "content": "You are a helpful mental health assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    reply = chat.choices[0].message.content

    return reply


# 🔥 REAL-TIME CHAT LOOP
if __name__ == "__main__":
    print("🧠 AI Mental Health Bot (type 'exit' to stop)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Take care ❤️")
            break

        response = generate_response(user_input)

        print("Bot:", response)