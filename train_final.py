import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("final_dataset.csv")

X = df[["packet_size", "protocol", "time_diff", "packet_rate"]]
y = df["label"]

label_map = {
    0: 0,
    1: 1,
    3: 2,
    4: 3
}

y = y.map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "xgboost_multiclass_ids_model.pkl")
print("\nModel saved as xgboost_multiclass_ids_model.pkl")
