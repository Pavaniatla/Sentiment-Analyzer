import os
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from imblearn.over_sampling import SMOTE


# Clean text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Map numeric ratings to sentiment classes
def map_rating_to_sentiment(rating):
    try:
        r = int(round(float(rating)))
        return f"class_{min(max(r, 1), 5)}"
    except:
        return None


# Load and preprocess data
def load_data():
    df = pd.read_csv("flipkart_product.csv", encoding='ISO-8859-1')[:15000]
    df.dropna(subset=['Summary', 'Rate'], inplace=True)
    df['sentiment'] = df['Rate'].apply(map_rating_to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)
    df['text'] = df['Summary'].apply(clean_text)
    print("✅ Original distribution:\n", df['sentiment'].value_counts())
    return df


# Tokenize text and encode labels
def preprocess(df, max_len=100):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=max_len, padding='post')
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df['sentiment']), num_classes=5)
    vocab_size = len(tokenizer.word_index) + 1
    return X, y, tokenizer, label_encoder, vocab_size


# Balance using SMOTE
def balance_data(X, y):
    y_labels = np.argmax(y, axis=1)
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y_labels)
    y_bal = to_categorical(y_bal, num_classes=5)
    return X_bal, y_bal


# Build the BiLSTM model
def build_bilstm_model(input_len, vocab_size, num_classes=5):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=input_len))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def train():
    df = load_data()
    X, y, tokenizer, label_encoder, vocab_size = preprocess(df, max_len=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = balance_data(X_train, y_train)

    model = build_bilstm_model(input_len=100, vocab_size=vocab_size, num_classes=5)
    stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=1, batch_size=32, callbacks=[stop])
    os.makedirs("models", exist_ok=True)
    model.save_weights("models/BiLSTM_weights.weights.h5")
    print(" BiLSTM model weights saved.")
    with open("models/BiLSTM_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("models/BiLSTM_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open("models/BiLSTM_vocab_size.pkl", "wb") as f:
        pickle.dump(vocab_size, f)
    print(" Tokenizer, label encoder, and vocab size saved.")

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
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    train()
