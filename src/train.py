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
    print(f"Поточна робоча директорія: {os.getcwd()}")
    
    # 1. Завантаження даних
    data_path = "data/prepared/train.csv"
    if not os.path.exists(data_path):
        print(f"ПОМИЛКА: Файл {data_path} не знайдено!")
        return

    train = pd.read_csv(data_path)
    print("Дані успішно завантажені.")
    
    X = train.select_dtypes(include=[np.number])
    if "RainTomorrow" in train.columns:
        y = train["RainTomorrow"].map({'No': 0, 'Yes': 1})
        if "RainTomorrow" in X.columns:
            X = X.drop("RainTomorrow", axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2. Тренування
    model = RandomForestClassifier(n_estimators=138, max_depth=15, random_state=42)
    model.fit(X, y)
    print("Модель натренована.")

    # 3. Оцінка та збереження метрик у корінь
    preds = model.predict(X)
    metrics = {
        "f1": float(f1_score(y, preds)),
        "accuracy": float(accuracy_score(y, preds))
    }
    
    # ПРИМУСОВИЙ ЗАПИС У КОРІНЬ
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Файл metrics.json створено.")

    # 4. Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")
    print("Файл confusion_matrix.png створено.")

    # 5. Збереження моделі
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Файл models/model.pkl створено.")

if __name__ == "__main__":
    train_final_model()