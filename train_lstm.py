# import os
# import pandas as pd
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.utils import to_categorical
# from imblearn.over_sampling import SMOTE

# # Cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text)
#     text = re.sub(r'\@\w+|\#','', text)
#     text = re.sub(r"[^a-zA-Z\s]", '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Adjust rating to 5 classes
# def map_rating_to_sentiment(rating):
#     return f"class_{int(rating)}"

# def load_dataset(file_path="GPT_reviews.csv"):
#     df = pd.read_csv(file_path)
#     df.dropna(subset=['Comment', 'Rating'], inplace=True)
#     df['sentiment'] = df['Rating'].apply(map_rating_to_sentiment)
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
#     y = to_categorical(y, num_classes=5)  # 5 classes now

#     return X, y, tokenizer, label_encoder

# def balance_data_with_smote(X, y):
#     smote = SMOTE(random_state=42)
#     X_res, y_res = smote.fit_resample(X, y)
#     print(f"Class distribution after SMOTE:\n{pd.Series(np.argmax(y_res, axis=1)).value_counts()}")
#     return X_res, y_res

# def build_model(input_len, model_type='LSTM', vocab_size=5000, num_classes=5):
#     model = Sequential()
#     model.add(Embedding(vocab_size, 128, input_length=input_len))
#     if model_type == 'RNN':
#         from tensorflow.keras.layers import SimpleRNN
#         model.add(SimpleRNN(128))
#     elif model_type == 'LSTM':
#         model.add(LSTM(128))
#     elif model_type == 'BiLSTM':
#         from tensorflow.keras.layers import Bidirectional
#         model.add(Bidirectional(LSTM(128)))
#     elif model_type == 'GRU':
#         from tensorflow.keras.layers import GRU
#         model.add(GRU(128))
#     else:
#         raise ValueError("Invalid model_type")
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# def train_model(X, y, model_type='LSTM', epochs=5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     print(f"Total samples: {len(X)}")
#     print(f"Training samples: {X_train.shape[0]}")
#     print(f"Testing samples: {X_test.shape[0]}")

#     X_train_res, y_train_res = balance_data_with_smote(X_train, y_train)
#     model = build_model(X.shape[1], model_type=model_type, num_classes=5)
#     history = model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    
#     # Save model weights
#     os.makedirs("models", exist_ok=True)

#     weights_path = f"models/{model_type}_weights.weights.h5"
#     model.save_weights(weights_path)
#     print(f"✅ Weights saved to {weights_path}")

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
#     plt.figure(figsize=(6,5))
#     sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()

# # =========================
# # MAIN EXECUTION
# # =========================

# if __name__ == "__main__":
#     data = load_dataset("GPT_reviews.csv")
#     data['text'] = data['Comment'].apply(clean_text)
#     X, y, tokenizer, label_encoder = preprocess_data(data)

#     # Save tokenizer and label encoder BEFORE training model
#     os.makedirs("models", exist_ok=True)
#     tokenizer_path = f"models/LSTM_tokenizer.pkl"
#     label_encoder_path = f"models/LSTM_label_encoder.pkl"

#     with open(tokenizer_path, "wb") as f:
#         pickle.dump(tokenizer, f)
#     print(f"✅ Tokenizer saved to {tokenizer_path}")

#     with open(label_encoder_path, "wb") as f:
#         pickle.dump(label_encoder, f)
#     print(f"✅ Label encoder saved to {label_encoder_path}")

#     model, history, X_test, y_test = train_model(X, y, model_type='LSTM', epochs=5)
#     evaluate_model(model, history, X_test, y_test, label_encoder)

#     while True:
#         test_input = input("\nEnter review (or 'exit'): ")
#         if test_input.lower() == 'exit':
#             break
#         print("Predicted Sentiment:", predict_sentiment(model, tokenizer, label_encoder, test_input))
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Map rating to 5 classes
def map_rating_to_sentiment(rating):
    try:
        r = float(rating)
        r = int(round(r))
        if r < 1:
            r = 1
        elif r > 5:
            r = 5
        return f"class_{r}"
    except:
        return None  # or some default class or drop row later


def load_dataset(file_path="flipkart_product.csv"):
    df = pd.read_csv(file_path, encoding='latin1')  # or try cp1252
    # df = df[:15000]
    df.dropna(subset=['Summary', 'Rate'], inplace=True)
    df['sentiment'] = df['Rate'].apply(map_rating_to_sentiment)
    df = df.dropna(subset=['sentiment'])  # remove rows where mapping failed

    print("Sentiment distribution before balancing:")
    print(df['sentiment'].value_counts())
    return df


def preprocess_data(data, max_len=100, num_words=5000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['text'])
    X = tokenizer.texts_to_sequences(data['text'])
    X = pad_sequences(X, maxlen=max_len, padding='post')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['sentiment'])
    y = to_categorical(y, num_classes=5)  # 5 classes

    return X, y, tokenizer, label_encoder

def balance_data_combined(X, y):
    """
    Combine undersampling + oversampling (SMOTE)
    Steps:
    - Undersample majority classes to reduce their count to some threshold
    - Oversample minority classes using SMOTE
    """
    print("Original class distribution:", Counter(np.argmax(y, axis=1)))
    
    # Convert one-hot y back to single label array for sampling
    y_labels = np.argmax(y, axis=1)
    
    # Undersample majority classes to threshold
    # Set threshold: e.g. median class count or fixed number
    target_count = 20000  # you can adjust this number based on your data
    
    rus = RandomUnderSampler(sampling_strategy={cls: min(count, target_count) 
                                                for cls, count in Counter(y_labels).items()},
                             random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y_labels)
    print("After undersampling:", Counter(y_rus))
    
    # Now apply SMOTE to balance minority classes
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = smote.fit_resample(X_rus, y_rus)
    print("After SMOTE oversampling:", Counter(y_res))
    
    # Convert labels back to one-hot
    y_res_ohe = to_categorical(y_res, num_classes=5)
    
    return X_res, y_res_ohe

def build_lstm_model(input_len, vocab_size=5000, num_classes=5):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=input_len))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(X, y, epochs=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Apply combined balancing on training data
    X_train_bal, y_train_bal = balance_data_combined(X_train, y_train)

    model = build_lstm_model(X.shape[1], vocab_size=5000, num_classes=5)
    history = model.fit(X_train_bal, y_train_bal, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    
    # Save model weights
    os.makedirs("models", exist_ok=True)
    weights_path = f"models/LSTM_combined_balancing.weights.h5"
    model.save_weights(weights_path)
    print(f"✅ Weights saved to {weights_path}")

    return model, history, X_test, y_test

def predict_sentiment(model, tokenizer, label_encoder, sentence, max_len=100):
    sentence = clean_text(sentence)
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    label = np.argmax(pred, axis=1)
    return label_encoder.inverse_transform(label)[0]

def evaluate_model(model, history, X_test, y_test, label_encoder):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ====== MAIN ======
if __name__ == "__main__":
    data = load_dataset("flipkart_product.csv")
    data['text'] = data['Summary'].apply(clean_text)
    X, y, tokenizer, label_encoder = preprocess_data(data)

    # Save tokenizer and label encoder BEFORE training model
    os.makedirs("models", exist_ok=True)
    tokenizer_path = "models/LSTM_tokenizer.pkl"
    label_encoder_path = "models/LSTM_label_encoder.pkl"

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved to {tokenizer_path}")

    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"✅ Label encoder saved to {label_encoder_path}")

    model, history, X_test, y_test = train_model(X, y, epochs=5)
    evaluate_model(model, history, X_test, y_test, label_encoder)

    while True:
        test_input = input("\nEnter review (or 'exit'): ")
        if test_input.lower() == 'exit':
            break
        pred = predict_sentiment(model, tokenizer, label_encoder, test_input)
        print(f"Predicted Sentiment: {pred}")
