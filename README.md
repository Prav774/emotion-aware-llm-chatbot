# AI-Powered Mental Health Support Assistant

An intelligent mental health support chatbot that combines a custom Deep Learning sentiment analysis model with Large Language Models (LLMs) to provide emotionally aware and context-sensitive conversations.

The system first analyzes user emotions using a custom LSTM-Attention neural network and then generates empathetic responses using an LLM, creating a hybrid AI architecture that goes beyond traditional chatbot implementations.

---

# Overview

Mental health support systems require more than simple question-answering capabilities. Understanding emotional context is critical for generating meaningful responses.

This project integrates:

* Deep Learning based sentiment analysis
* Emotion-aware response generation
* Conversation memory
* Emotional shift detection
* Real-time web interaction

The chatbot analyzes a user's emotional state before generating a response, allowing conversations to adapt based on emotional changes throughout the session.

---

# Key Features

### Emotion-Aware Conversations

The chatbot predicts user sentiment before generating responses.

Example:

User:

> I failed my exam and feel terrible.

Detected Sentiment:

> Negative

Generated Response:

> I'm sorry you're going through this. Failing an exam can be difficult, but it doesn't define your abilities or future potential.

---

### LSTM-Attention Sentiment Analysis

A custom Deep Learning model built using:

* Embedding Layer
* LSTM Networks
* Attention Mechanism
* Dense Layers

The attention mechanism helps the model focus on emotionally significant words within user messages.

---

### Conversation Memory

The chatbot maintains recent conversation history to preserve context.

Benefits:

* Improved conversational flow
* Better understanding of ongoing discussions
* More natural responses

---

### Emotional Shift Detection

The system tracks changes in sentiment across messages.

Example:

Message 1:

> Today was amazing.

Detected:

> Positive

Message 2:

> But now I'm feeling hopeless.

Detected:

> Negative

The chatbot recognizes this emotional shift and adapts its responses accordingly.

---

### Real-Time Web Interface

Users interact with the chatbot through a Flask-powered web application.

Features:

* Interactive chat interface
* Instant sentiment analysis
* Real-time response generation
* Simple and lightweight deployment

---

# System Architecture

```text
User Input
     │
     ▼
Text Preprocessing
     │
     ▼
Tokenizer
     │
     ▼
LSTM + Attention Model
     │
     ▼
Sentiment Prediction
     │
     ▼
Conversation Memory
     │
     ▼
Groq LLM
     │
     ▼
Empathetic Response
     │
     ▼
Web Interface
```

---

# Technology Stack

## Backend

* Python
* Flask

## Machine Learning

* TensorFlow
* Keras
* LSTM Networks
* Attention Mechanism

## Natural Language Processing

* NLTK
* Tokenization
* Text Cleaning

## Frontend

* HTML
* CSS
* JavaScript

## AI Services

* Groq API

---

# Project Structure

```text
MentalHealthBot/
│
├── backend_python/
│   ├── app.py
│   ├── chatbot.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── tokenizer.py
│   ├── train_sentiment.py
│   ├── sentiment_model.keras
│   ├── tokenizer.pkl
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── __pycache__/
│
├── dataset/
│   └── Meld/
│       └── train_sent_emo.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Dataset

This project uses emotion and sentiment-based conversational datasets for training and evaluation.

Dataset characteristics:

* Human conversations
* Emotional labels
* Sentiment categories
* Real-world dialogue examples

The dataset helps the model learn emotional patterns in human communication.

---

# Model Architecture

## Input Layer

Converts user text into numerical representations.

## Embedding Layer

Transforms words into dense vector representations.

## LSTM Layer

Captures sequential dependencies and contextual information.

## Attention Layer

Highlights emotionally important words within sentences.

## Dense Layers

Generate sentiment predictions.

## Output Layer

Produces sentiment classification results.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Prav774/emotion-aware-llm-chatbot.git
cd MentalHealthBot
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Running the Application

Start the Flask server:

```bash
python backend_python/app.py
```

Open your browser:

```text
http://localhost:5000
```

---

# Training the Model

To retrain the sentiment analysis model:

```bash
python backend_python/train_sentiment.py
```

This will:

1. Load the dataset
2. Preprocess text
3. Train the LSTM-Attention model
4. Save the tokenizer
5. Save the trained model

Generated files:

```text
sentiment_model.keras
tokenizer.pkl
```

---

# Example Workflow

### User Input

```text
I feel stressed and overwhelmed lately.
```

### Sentiment Prediction

```text
Negative
```

### Response Generation

```text
It sounds like you've been carrying a lot recently.
Would you like to talk about what's been causing the stress?
```

---

# Future Improvements

Planned enhancements include:

* Multi-class emotion detection
* Anxiety recognition
* Depression risk assessment
* Crisis and self-harm detection
* User authentication
* Database integration
* Mobile application
* Dashboard analytics
* Speech-to-text support
* Voice-based conversations

---

# Learning Outcomes

This project demonstrates practical experience in:

* Deep Learning
* Natural Language Processing
* Sentiment Analysis
* Attention Mechanisms
* LLM Integration
* Flask Development
* AI System Design
* Human-Centered AI

---

# Disclaimer

This application is intended for educational and research purposes only.

It is not a replacement for professional medical advice, diagnosis, therapy, or mental health treatment. Individuals experiencing severe emotional distress should seek assistance from qualified healthcare professionals or emergency services.

---

# Author

Developed as an AI and NLP project focused on emotion-aware conversational systems using Deep Learning and Large Language Models.
