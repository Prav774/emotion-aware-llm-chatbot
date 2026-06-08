import pandas as pd
import pickle

from preprocess import clean_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from model import build_model

# 🔹 load dataset
df = pd.read_csv("../dataset/sentiment140.csv", encoding='latin-1', header=None)

# 🔹 assign column names
df.columns = ["target", "id", "date", "flag", "user", "text"]

# 🔹 keep only needed columns
df = df[["target", "text"]]

# 🔹 convert labels (4 → 1)
df['target'] = df['target'].replace(4, 1)

# 🔹 take subset (for speed)
df = df.sample(10000, random_state=42)

# 🔹 clean text
df['text'] = df['text'].apply(clean_text)

print(df.head())

# 🔹 tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

# 🔹 save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# 🔹 convert text → sequences
sequences = tokenizer.texts_to_sequences(df['text'])

# 🔹 padding (IMPORTANT: maxlen = 50)
MAX_LEN = 50
X = pad_sequences(sequences, maxlen=MAX_LEN)

# 🔹 labels (categorical)
y = to_categorical(df['target'])

print("X shape:", X.shape)
print("y shape:", y.shape)

# 🔹 build model
model = build_model()

# 🔹 train model
model.fit(X, y, epochs=3, batch_size=64, validation_split=0.2)

# 🔹 save model
model.save("sentiment_model.keras")

print("✅ Training complete + model & tokenizer saved")