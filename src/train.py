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
    # 1. Завантаження даних
    # Перевіряємо різні можливі шляхи для надійності в CI
    data_path = "data/prepared/train.csv"
    
    if not os.path.exists(data_path):
        print(f"КРИТИЧНА ПОМИЛКА: Файл {data_path} не знайдено!")
        # Виводимо список файлів для відладки
        print("Вміст папки data/prepared/:", os.listdir("data/prepared/") if os.path.exists("data/prepared/") else "папка відсутня")
        return

    print(f"Завантаження даних з {data_path}...")
    train = pd.read_csv(data_path)
    
    # Підготовка ознак та цільової змінної
    X = train.select_dtypes(include=[np.number])
    if "RainTomorrow" in train.columns:
        y = train["RainTomorrow"].map({'No': 0, 'Yes': 1})
        if "RainTomorrow" in X.columns:
            X = X.drop("RainTomorrow", axis=1)
    
    # Заповнюємо пропуски
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 2. Параметри моделі (згідно з вашим найкращим Trial)
    best_params = {
        "n_estimators": 138,
        "max_depth": 15,
        "random_state": 42
    }

    print("Тренування моделі RandomForest...")
    model = RandomForestClassifier(**best_params)
    model.fit(X, y)

    # 3. Оцінка
    preds = model.predict(X)
    f1 = f1_score(y, preds)
    acc = accuracy_score(y, preds)

    # 4. Збереження артефактів (ВИПРАВЛЕНО ШЛЯХИ)
    # Зберігаємо в корінь (.), щоб pytest їх одразу побачив
    metrics = {
        "f1": float(f1),
        "accuracy": float(acc)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Метрики збережено: F1={f1:.4f}")

    # Створення Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    print("Графік збережено в confusion_matrix.png")

    # Збереження моделі
    os.makedirs("models", exist_ok=True)
    model_file = "models/model.pkl"
    joblib.dump(model, model_file)
    print(f"Модель збережено в {model_file}")

if __name__ == "__main__":
    train_final_model()