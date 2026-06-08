import re
import string
from nltk.corpus import stopwords

# load stopwords
stop_words = set(stopwords.words('english'))

# 🔥 KEEP IMPORTANT NEGATIONS
negations = {"not", "no", "never", "n't"}
stop_words = stop_words - negations


def clean_text(text):
    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize
    words = text.split()

    # remove stopwords (but keep negations)
    words = [word for word in words if word not in stop_words]

    # join back
    return " ".join(words)