import pickle
import numpy as np

from preprocess import clean_text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50


def predict(text):
    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(padded)[0]
    label = np.argmax(pred)

    return "positive" if label == 1 else "negative"