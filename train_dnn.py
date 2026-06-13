import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("final_dataset.csv")

X = df[["packet_size", "protocol", "time_diff", "packet_rate"]]
y = df["label"]

# Clean missing/infinite values
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)

# Remap labels
label_map = {
    0: 0,   # Normal
    1: 1,   # Flooding/Portscan
    3: 2,   # Blackhole
    4: 3    # Grayhole
}
y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save("dnn_ids_model.keras")
joblib.dump(scaler, "dnn_scaler.pkl")

print("\nModel saved as dnn_ids_model.keras")
print("Scaler saved as dnn_scaler.pkl")
