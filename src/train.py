import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import os

def train_final_model():
    # 1. Завантаження даних (використовуємо підготовлені дані з ЛР2)
    data_path = "data/prepared/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    train = pd.read_csv(data_path)
    
    # Видаляємо нечислові колонки (як у ЛР3)
    X = train.select_dtypes(include=[np.number])
    if "RainTomorrow" in train.columns:
        y = train["RainTomorrow"].map({'No': 0, 'Yes': 1})
        if "RainTomorrow" in X.columns:
            X = X.drop("RainTomorrow", axis=1)
    
    # Заповнюємо пропуски
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2. Використання найкращих параметрів (отриманих в ЛР3)
    # n_estimators=138, max_depth=15 (згідно з вашим Trial 13)
    best_params = {
        "n_estimators": 138,
        "max_depth": 15,
        "random_state": 42
    }

    model = RandomForestClassifier(**best_params)
    model.fit(X, y)

    # 3. Оцінка моделі
    preds = model.predict(X)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)

    # 4. Збереження метрик у JSON (для CML та Quality Gate) [cite: 767-769]
    metrics = {
        "f1": float(f1),
        "accuracy": float(acc)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

    # 5. Створення та збереження Confusion Matrix [cite: 766]
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    print("Visualization saved to confusion_matrix.png")

    # 6. Збереження моделі [cite: 763]
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_final_model()