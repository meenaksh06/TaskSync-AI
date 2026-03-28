# train_and_predict.py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# ------------------------
# 1. Load audio and extract features
# ------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# ------------------------
# 2. Prepare dataset
# ------------------------
def create_dataset(data_dir):
    features = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            label = file.split(".")[0]  # e.g., hello.wav → hello
            path = os.path.join(data_dir, file)
            feat = extract_features(path)
            features.append(feat)
            labels.append(label)
    X = np.array(features)
    y = np.array(labels)
    return X, y

# ------------------------
# 3. Train models
# ------------------------
def train_models(X, y):
    # Encode labels numerically
    label_to_num = {label: i for i, label in enumerate(set(y))}
    num_to_label = {i: label for label, i in label_to_num.items()}
    y_encoded = np.array([label_to_num[l] for l in y])

    # Train Logistic Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Train Linear Regression (to predict encoded label as number)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = np.round(lin_reg.predict(X_test))
    mse = mean_squared_error(y_test, y_pred_lin)

    print(f"[Metrics] Logistic Accuracy: {acc:.2f}, Linear MSE: {mse:.4f}")
    return log_reg, lin_reg, num_to_label

# ------------------------
# 4. Predict speech file
# ------------------------
def predict(file_path, log_model, lin_model, num_to_label):
    feat = extract_features(file_path).reshape(1, -1)
    pred_log = int(log_model.predict(feat)[0])
    pred_lin = int(np.round(lin_model.predict(feat)[0]))

    label_log = num_to_label.get(pred_log, "unknown")
    label_lin = num_to_label.get(pred_lin, "unknown")

    print(f"[Speech-to-Text] Logistic says: '{label_log}' | Linear says: '{label_lin}'")

# ------------------------
# 5. Run
# ------------------------
if __name__ == "__main__":
    data_dir = "data"
    X, y = create_dataset(data_dir)
    log_model, lin_model, num_to_label = train_models(X, y)

    # Test on a new sample (or existing one)
    test_file = "data/test.wav"  # Record a test word
    if os.path.exists(test_file):
        predict(test_file, log_model, lin_model, num_to_label)
    else:
        print("[WARN] No test.wav found in data/. Record one to test.")
