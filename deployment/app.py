from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nothing', 'nowhere'}

current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(current_dir, "models", "sentiment_lstm_model_acc_0.92.h5")
TOKENIZER_PATH = os.path.join(current_dir, "models", "tokenizer_acc_0.92.pkl")
LABEL_ENCODER_PATH = os.path.join(current_dir, "models", "label_encoder_acc_0.92.pkl")

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

max_len = 80

def clean_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Please provide 'text'"}), 400

    text = data["text"]
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad_seq)
    label_index = np.argmax(pred, axis=1)
    sentiment = label_encoder.inverse_transform(label_index)[0]

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": float(np.max(pred))
    })

if __name__ == "__main__":
    app.run(debug=True)
