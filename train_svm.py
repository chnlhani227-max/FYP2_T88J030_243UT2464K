import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("final_dataset.csv")

X = df[["packet_size", "protocol", "time_diff", "packet_rate"]]
y = df["label"]

# Clean missing or infinite values
X = X.replace([float("inf"), float("-inf")], 0)
X = X.fillna(0)

label_map = {
    0: 0,
    1: 1,
    3: 2,
    4: 3
}

y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced"))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "svm_ids_model.pkl")

print("\nModel saved as svm_ids_model.pkl")
