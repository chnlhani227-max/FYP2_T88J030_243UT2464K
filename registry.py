from core.config_user import CUSTOM_ALGOS

# Base Library
FULL_ALGO_MAP = {
    # Core Requirements
    "RandomForest": "sklearn_tree",
    "SVM": "sklearn_linear",
    "DNN (Dense)": "dnn",
    "TCN": "cnn",
    "LSTM": "rnn",
    "Encoder": "dnn",
    # Extended Library
    "KNN": "sklearn_linear",
    "DecisionTree": "sklearn_tree",
    "NaiveBayes": "sklearn_linear",
    "MobileNetV3": "cnn",
    "ResNet-50": "cnn",
    "FedAvg": "fl_proxy",
    "DQN": "dnn"
}

# Inject User Custom Algos (Default to Tree logic if unknown)
for algo in CUSTOM_ALGOS:
    if algo not in FULL_ALGO_MAP:
        FULL_ALGO_MAP[algo] = "sklearn_tree"

CATEGORIES = {
    "My Selected Algorithms": CUSTOM_ALGOS,
    "Core Requirements": ["RandomForest", "SVM", "DNN (Dense)", "TCN", "LSTM", "Encoder"],
    "Full Library": list(FULL_ALGO_MAP.keys())
}
