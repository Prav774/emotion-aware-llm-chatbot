from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def create_tokenizer(texts, max_words=10000, max_len=50):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    # save tokenizer (IMPORTANT for later chatbot use)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return padded, tokenizer