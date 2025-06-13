# bilstm_sentiment.py

import os, re, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    return re.sub(r'\s+', ' ', text).strip()

def map_rating_to_sentiment(rating):
    try:
        r = int(round(float(rating)))
        return f"class_{min(max(r, 1), 5)}"
    except:
        return None

def load_dataset(path="GPT_reviews.csv"):
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')[:15000]
    df.dropna(subset=['Comment', 'Rating'], inplace=True)
    df['sentiment'] = df['Rating'].apply(map_rating_to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)
    print("Sentiment distribution before balancing:\n", df['sentiment'].value_counts())
    return df

def preprocess_data(data, max_len=200, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['text'])
    X = pad_sequences(tokenizer.texts_to_sequences(data['text']), maxlen=max_len, padding='post')
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(data['sentiment']), num_classes=5)
    return X, y, tokenizer, label_encoder

def balance_data_with_smote(X, y):
    y_labels = np.argmax(y, axis=1)
    print("Original:", Counter(y_labels))
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y_labels)
    print("After SMOTE:", Counter(y_res))
    return X_res, to_categorical(y_res, num_classes=y.shape[1])

def build_bilstm_model(input_len, vocab_size=10000, num_classes=5):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=input_len))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(X, y, epochs=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    X_train_res, y_train_res = balance_data_with_smote(X_train, y_train)
    model = build_bilstm_model(X.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[early_stop])
    os.makedirs("models", exist_ok=True)
    model.save_weights("models/BiLSTM_smote.weights.h5")
    print("✅ Weights saved")
    return model, history, X_test, y_test

def evaluate_model(model, history, X_test, y_test, label_encoder):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val'); plt.legend(); plt.title("Accuracy")
    plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val'); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.show()
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True,
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix"); plt.show()

def predict_sentiment(model, tokenizer, label_encoder, sentence, max_len=200):
    sequence = tokenizer.texts_to_sequences([clean_text(sentence)])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    pred = np.argmax(model.predict(padded), axis=1)
    return label_encoder.inverse_transform(pred)[0]

if __name__ == "__main__":
    df = load_dataset()
    df['text'] = df['Comment'].apply(clean_text)
    X, y, tokenizer, label_encoder = preprocess_data(df)
    with open("models/BiLSTM_tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
    with open("models/BiLSTM_label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)
    print("✅ Tokenizer and LabelEncoder saved")
    model, history, X_test, y_test = train_model(X, y)
    evaluate_model(model, history, X_test, y_test, label_encoder)
    while True:
        text = input("Enter review (or 'exit'): ")
        if text.lower() == 'exit': break
        print("Predicted:", predict_sentiment(model, tokenizer, label_encoder, text))
