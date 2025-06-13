# import os
# import re
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils import resample

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
# from tensorflow.keras.utils import to_categorical

# from imblearn.over_sampling import SMOTE
# from collections import Counter

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r'\@\w+|\#', '', text)
#     text = re.sub(r"[^a-zA-Z\s]", '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def map_rating_to_sentiment(rating):
#     mapping = {
#         1: "class_1",
#         2: "class_2",
#         3: "class_3",
#         4: "class_4",
#         5: "class_5",
#     }
#     return mapping.get(int(rating), "class_3")

# def load_dataset(file_path="flipkart_product.csv"):
#     df = pd.read_csv(file_path, encoding='ISO-8859-1')
#     df = df[:15000]
#     df.dropna(subset=['Review', 'Rate'], inplace=True)
#     df['sentiment'] = df['Rate'].apply(map_rating_to_sentiment)
#     print("Sentiment distribution before balancing:")
#     print(df['sentiment'].value_counts())
#     return df

# def preprocess_data(data, max_len=100, num_words=5000):
#     tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
#     tokenizer.fit_on_texts(data['text'])
#     X = tokenizer.texts_to_sequences(data['text'])
#     X = pad_sequences(X, maxlen=max_len, padding='post')

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(data['sentiment'])
#     y = to_categorical(y, num_classes=len(label_encoder.classes_))

#     return X, y, tokenizer, label_encoder

# def balance_data(X, y, strategy="combo"):
#     if strategy == "undersample":
#         df_X = pd.DataFrame(X)
#         df_y = pd.DataFrame(y, columns=[f"class_{i}" for i in range(y.shape[1])])
#         df_y['label'] = df_y.idxmax(axis=1)
#         min_count = df_y['label'].value_counts().min()
#         balanced_df = pd.concat([
#             resample(df_y[df_y['label'] == label], replace=False, n_samples=min_count, random_state=42)
#             for label in df_y['label'].unique()
#         ])
#         balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

#         X_bal = X[balanced_df.index]
#         y_bal = y[balanced_df.index]

#     elif strategy == "smote":
#         smote = SMOTE(random_state=42)
#         X_flat = X.reshape((X.shape[0], -1))
#         y_labels = np.argmax(y, axis=1)
#         X_res, y_res = smote.fit_resample(X_flat, y_labels)
#         X_bal = X_res.reshape((-1, X.shape[1]))
#         y_bal = to_categorical(y_res, num_classes=y.shape[1])

#     elif strategy == "combo":
#         # Step 1: Undersample majority classes to upper limit
#         df_X = pd.DataFrame(X)
#         df_y = pd.DataFrame(y, columns=[f"class_{i}" for i in range(y.shape[1])])
#         df_y['label'] = df_y.idxmax(axis=1)

#         upper_limit = 1500  # You can tune this number

#         dfs = []
#         for label in df_y['label'].unique():
#             df_label = df_y[df_y['label'] == label]
#             if len(df_label) > upper_limit:
#                 df_downsampled = resample(df_label, replace=False, n_samples=upper_limit, random_state=42)
#             else:
#                 df_downsampled = df_label
#             dfs.append(df_downsampled)

#         df_combined = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
#         X_combined = X[df_combined.index]
#         y_combined = y[df_combined.index]

#         # Step 2: Apply SMOTE to balance classes
#         smote = SMOTE(random_state=42)
#         X_flat = X_combined.reshape((X_combined.shape[0], -1))
#         y_labels = np.argmax(y_combined, axis=1)
#         X_res, y_res = smote.fit_resample(X_flat, y_labels)

#         X_bal = X_res.reshape((-1, X.shape[1]))
#         y_bal = to_categorical(y_res, num_classes=y.shape[1])

#     else:
#         return X, y

#     print(f"Balanced class distribution with {strategy}:")
#     print(Counter(np.argmax(y_bal, axis=1)))

#     return X_bal, y_bal

# def build_model(input_len, vocab_size=5000, num_classes=5):
#     model = Sequential()
#     model.add(Embedding(vocab_size, 128, input_length=input_len))
#     model.add(SimpleRNN(128))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# def train_model(X, y, tokenizer, label_encoder, strategy="combo", epochs=5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Training samples before balancing: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

#     X_train_res, y_train_res = balance_data(X_train, y_train, strategy)

#     print(f"Training samples after balancing: {X_train_res.shape[0]}")

#     model = build_model(X.shape[1], vocab_size=len(tokenizer.word_index) + 1, num_classes=y.shape[1])
#     history = model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

#     os.makedirs("models", exist_ok=True)
#     model.save_weights("models/rnn_weights.weights.h5")
#     with open("models/rnn_tokenizer.pkl", "wb") as f:
#         pickle.dump(tokenizer, f)
#     with open("models/rnn_label_encoder.pkl", "wb") as f:
#         pickle.dump(label_encoder, f)

#     return model, history, X_test, y_test

# def predict_sentiment(model, tokenizer, label_encoder, sentence, max_len=100):
#     sentence = clean_text(sentence)
#     seq = tokenizer.texts_to_sequences([sentence])
#     padded = pad_sequences(seq, maxlen=max_len, padding='post')
#     pred = model.predict(padded)
#     label = np.argmax(pred, axis=1)
#     return label_encoder.inverse_transform(label)[0]

# def evaluate_model(model, history, X_test, y_test, label_encoder):
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     y_pred_probs = model.predict(X_test)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     y_true = np.argmax(y_test, axis=1)

#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_, cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

# if __name__ == "__main__":
#     data = load_dataset("flipkart_product.csv")
#     data['text'] = data['Review'].apply(clean_text)
#     X, y, tokenizer, label_encoder = preprocess_data(data)

#     # Choose balancing strategy: "combo" = undersampling + SMOTE
#     model, history, X_test, y_test = train_model(X, y, tokenizer, label_encoder, strategy="combo", epochs=5)

#     evaluate_model(model, history, X_test, y_test, label_encoder)

#     while True:
#         test_input = input("\nEnter review (or 'exit'): ")
#         if test_input.lower() == 'exit':
#             break
#         print("Predicted Sentiment:", predict_sentiment(model, tokenizer, label_encoder, test_input))
import os, re, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_rating_to_sentiment(rating):
    try:
        r = int(round(float(rating)))
        return f"class_{min(max(r, 1), 5)}"
    except:
        return None

def load_data():
    df = pd.read_csv("flipkart_product.csv",encoding='ISO-8859-1')[:15000]
    df.dropna(subset=['Summary', 'Rate'], inplace=True)
    df['sentiment'] = df['Rate'].apply(map_rating_to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)
    df['text'] = df['Summary'].apply(clean_text)
    print("✅ Original distribution:\n", df['sentiment'].value_counts())
    return df

def preprocess(df, max_len=100, vocab_size=5000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    X = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(X, maxlen=max_len, padding='post')

    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    y = to_categorical(y, num_classes=5)

    return X, y, tokenizer, le



def balance_data(X, y):
    y_labels = np.argmax(y, axis=1)
    print("✅ Before balancing:", Counter(y_labels))

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y_labels)
    y_bal_onehot = to_categorical(y_bal, num_classes=5)

    print("✅ After balancing:", Counter(y_bal))
    return X_bal, y_bal_onehot


def build_model(input_len, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=input_len))
    model.add(SimpleRNN(128))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(X, y, epochs=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = balance_data(X_train, y_train)

    model = build_model(X.shape[1])
    stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                     validation_data=(X_test, y_test), callbacks=[stop])

    os.makedirs("models", exist_ok=True)
    model.save_weights("models/RNN_combined.weights.h5")
    print("✅ Weights saved.")
    return model, hist, X_test, y_test

def evaluate(model, hist, X_test, y_test, label_encoder):
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.legend(), plt.title("Accuracy"), plt.show()

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix"), plt.show()

if __name__ == "__main__":
    df = load_data()
    X, y, tokenizer, label_encoder = preprocess(df)

    with open("models/RNN_tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
    with open("models/RNN_label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)

    model, hist, X_test, y_test = train(X, y)
    evaluate(model, hist, X_test, y_test, label_encoder)
