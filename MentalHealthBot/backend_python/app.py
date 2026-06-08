from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from groq import Groq
from collections import deque

from predict import predict

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found")

client = Groq(api_key=api_key)

# 🔥 memory stores (message, sentiment)
memory = deque(maxlen=5)


def generate_response(user_input):
    try:
        sentiment = predict(user_input)

        # store message + sentiment
        memory.append((user_input, sentiment))

        # 🔥 detect emotional shift
        shift = False
        if len(memory) >= 2:
            prev_sentiment = memory[-2][1]
            if prev_sentiment == "positive" and sentiment == "negative":
                shift = True

        # 🔥 NEW: structured conversation (IMPORTANT FIX)
        conversation = ""
        for msg, sent in memory:
            conversation += f"User: {msg} ({sent})\n"

        # 🔥 UPDATED PROMPT (CONTINUITY FIX)
        prompt = f"""
You are a mental health chatbot.

Conversation so far:
{conversation}

Current message:
User: {user_input}

Emotional shift detected: {shift}

STRICT RULES:
- Focus mainly on the CURRENT message
- Maintain continuity with previous messages
- Do NOT jump topics
- Do NOT bring unrelated past topics
- Keep response short (1–2 lines)
- Be emotionally consistent
- Do NOT repeat phrases

Reply:
"""

        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a calm and understanding assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        return chat.choices[0].message.content.strip()

    except Exception as e:
        print("❌ ERROR:", str(e))
        return "I'm here with you."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        if not user_input:
            return jsonify({"response": "Say something 🙂"})

        response = generate_response(user_input)

        return jsonify({"response": response})

    except Exception as e:
        print("❌ ROUTE ERROR:", str(e))
        return jsonify({"response": "Server error"})


if __name__ == "__main__":
    app.run(debug=True)