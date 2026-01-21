import pandas as pd
import numpy as np
import time, os, joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from core.registry import FULL_ALGO_MAP
from core.profiler import EdgeProfiler

class StudentTrainer:
    def __init__(self):
        self.models_dir = "deployment_builds"
        os.makedirs(self.models_dir, exist_ok=True)
        self.profiler = EdgeProfiler()

    def train_model(self, algo_name, data_path, device_name, hp_epochs, hp_batch, hp_est):
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        logic = FULL_ALGO_MAP.get(algo_name, "sklearn_tree")
        start = time.time()
        
        if logic in ["dnn", "cnn", "rnn"]:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=hp_epochs, batch_size=hp_batch, verbose=0)
            y_probs = model.predict(X_test, verbose=0)
            preds = (y_probs > 0.5).astype(int).flatten()
        else:
            model = RandomForestClassifier(n_estimators=hp_est)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        end = time.time()
        
        # Calculate Advanced Metrics for Hypotheses
        acc = accuracy_score(y_test, preds) * 100
        prec = precision_score(y_test, preds, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, preds, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, preds, average='weighted') * 100
        
        # FPR Calculation (Hypothesis 3)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel() if len(np.unique(y_test)) == 2 else (0,0,0,0)
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

        hw_metrics = self.profiler.profile_model(model, logic, X_train.shape, device_name)
        
        return {
            "Algorithm": algo_name,
            "Accuracy (%)": acc,
            "Precision (%)": prec,
            "Recall/Detection Rate (%)": rec,
            "F1-Score (%)": f1,
            "False Positive Rate (%)": fpr,
            "Training Time (ms)": (end - start) * 1000,
            **hw_metrics
        }
