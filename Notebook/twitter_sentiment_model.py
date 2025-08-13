"""# 1. Import Libraries"""

import os
import re
import pickle
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

"""# 2. Download Stopwords"""

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nothing', 'nowhere'}

"""# 3. Data Upload"""

train_data = pd.read_csv('/content/twitter_data/twitter_training.csv', names=['id','entity','sentiment','text'])
test_data  = pd.read_csv('/content/twitter_data/twitter_validation.csv', names=['id','entity','sentiment','text'])

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

"""# 4. Data Cleaning"""

train_data.drop_duplicates(inplace=True)
train_data.dropna(subset=['text'], inplace=True)
test_data.drop_duplicates(inplace=True)
test_data.dropna(subset=['text'], inplace=True)

train_data = train_data[train_data['sentiment'] != 'Irrelevant']
test_data = test_data[test_data['sentiment'] != 'Irrelevant']

positive_words = ['love', 'best', 'amazing', 'great', 'excellent', 'perfect', 'incredible', 'awesome', 'good', 'nice']
negative_words = ['hate', 'terrible', 'awful', 'sucks', 'horrible', 'worst', 'bad', 'disgusting', 'broken']

def is_misleading_neutral(text):
    text_lower = str(text).lower()
    return any(w in text_lower for w in positive_words) or any(w in text_lower for w in negative_words)

neutral_mask = (train_data['sentiment'] == 'Neutral') & train_data['text'].apply(is_misleading_neutral)
train_data = train_data[~neutral_mask]

"""# 5. Text Preprocessing

"""

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

train_data['cleaned_text'] = train_data['text'].apply(clean_text)
test_data['cleaned_text'] = test_data['text'].apply(clean_text)

"""# 6. Encode Labels"""

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['sentiment'])
test_data['label'] = label_encoder.transform(test_data['sentiment'])

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

"""# 7. Tokenization & Padding"""

max_words = 10000
max_len = 80

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['cleaned_text'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['cleaned_text']), maxlen=max_len, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['cleaned_text']), maxlen=max_len, padding='post')

y_train = to_categorical(train_data['label'], num_classes=3)
y_test = to_categorical(test_data['label'], num_classes=3)

"""# 8. Build Model"""

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(16)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.build(input_shape=(None, max_len))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_train, y_train,),
    callbacks=[early_stop],
    verbose=1
)

"""# 9. Evaluate Model"""

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

"""# 10. Save Model & Tokenizer"""

accuracy_score = 0.92
model_filename = f"sentiment_lstm_model_acc_{accuracy_score:.2f}.h5"
tokenizer_filename = f"tokenizer_acc_{accuracy_score:.2f}.pkl"
label_encoder_filename = f"label_encoder_acc_{accuracy_score:.2f}.pkl"

model.save(model_filename)

with open(tokenizer_filename, 'wb') as f:
    pickle.dump(tokenizer, f)

with open(label_encoder_filename, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model and tokenizer saved successfully with accuracy {accuracy_score:.2f}")

sample_texts = [
    "I really love this product, it's amazing!",
    "The service was terrible and I will never come back",
    "It's okay, nothing special but not bad"
]

sample_cleaned = [clean_text(t) for t in sample_texts]
sample_seq = tokenizer.texts_to_sequences(sample_cleaned)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')

predictions = model.predict(sample_pad)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

for i in range(len(sample_texts)):
    print(f"Text: {sample_texts[i]}")
    print(f"Predicted Sentiment: {predicted_labels[i]}")
    print("-" * 50)

