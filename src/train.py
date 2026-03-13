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
import sys

def train_final_model():
    try:
        print("--- СТАРТ ТРЕНУВАННЯ ---")
        
        # 1. Дані
        data_path = "data/prepared/train.csv"
        if not os.path.exists(data_path):
            print(f"ПОМИЛКА: {data_path} не знайдено!")
            sys.exit(1)

        train = pd.read_csv(data_path)
        X = train.select_dtypes(include=[np.number])
        if "RainTomorrow" in train.columns:
            y = train["RainTomorrow"].map({'No': 0, 'Yes': 1})
            if "RainTomorrow" in X.columns:
                X = X.drop("RainTomorrow", axis=1)
        
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # 2. Модель
        model = RandomForestClassifier(n_estimators=138, max_depth=15, random_state=42)
        model.fit(X, y)

        # 3. Збереження (в КОРІНЬ для тестів)
        preds = model.predict(X)
        metrics = {"f1": float(f1_score(y, preds)), "accuracy": float(accuracy_score(y, preds))}
        
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig("confusion_matrix.png")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        
        print("--- ВСІ АРТЕФАКТИ СТВОРЕНО УСПІШНО ---")
        
    except Exception as e:
        print(f"КРИТИЧНА ПОМИЛКА: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_final_model()