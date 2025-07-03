import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from model_utils import create_model


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", '', text)
    text = re.sub(r'\\@\\w+|\\#', '', text)
    text = re.sub(r"[^a-zA-Z\\s]", '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text


def map_rating_to_sentiment(rating):
    try:
        r = int(round(float(rating)))
        return f"class_{min(max(r, 1), 5)}"
    except:
        return None


def load_data():
    df = pd.read_csv("flipkart_product.csv", encoding='ISO-8859-1')[:15000]
    df.dropna(subset=['Summary', 'Rate'], inplace=True)
    df['sentiment'] = df['Rate'].apply(map_rating_to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)
    df['text'] = df['Summary'].apply(clean_text)
    print("\u2705 Original distribution:\n", df['sentiment'].value_counts())
    return df


def preprocess(df, max_len=100):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=max_len, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df['sentiment']), num_classes=5)
    return X, y, tokenizer, label_encoder, vocab_size


def balance_data(X, y):
    y_labels = np.argmax(y, axis=1)
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y_labels)
    y_bal = to_categorical(y_bal, num_classes=5)
    return X_bal, y_bal


def train():
    df = load_data()
    X, y, tokenizer, label_encoder, vocab_size = preprocess(df, max_len=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balance_data(X_train, y_train)

    model = create_model(input_len=100, vocab_size=vocab_size, model_type='GRU', num_classes=5)
    stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, callbacks=[stop])
    os.makedirs("models", exist_ok=True)
    model.save_weights("models/GRU_weights.weights.h5")
    print("\u2705 GRU model weights saved.")
    with open("models/GRU_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("models/GRU_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/GRU_vocab_size.pkl", "wb") as f:
        pickle.dump(vocab_size, f)
    print("\u2705 Tokenizer, label encoder, and vocab size saved.")

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.show()

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    train()
