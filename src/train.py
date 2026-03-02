import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Ініціалізація експерименту (Крок 4.9.4)
mlflow.set_experiment("Rain_Australia_Experiment")

# 2. Завантаження та підготовка даних (Крок 4.9.1-2)
# Видаляємо рядки, де немає цільової змінної
df = pd.read_csv('data/raw/weatherAUS.csv').dropna(subset=['RainTomorrow'])

# Вибираємо числові ознаки для базової моделі (Baseline)
cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm']
X = df[cols]
y = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Заповнюємо пропуски середнім значенням (Imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 3. Розділення на тренувальну та тестову вибірки (Крок 4.9.3)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 4. Виконання 5 експериментів (Крок 4.2.7)
# Ми будемо змінювати глибину дерева (max_depth)
for depth in [2, 5, 10, 15, 20]:
    with mlflow.start_run(run_name=f"Run_Depth_{depth}"):
        # Ініціалізація та навчання моделі
        model = RandomForestClassifier(max_depth=depth, n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Отримання прогнозів
        predictions = model.predict(X_test)
        
        # Розрахунок метрик (Крок 4.9.5)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # 5. Логування результатів у MLflow (Крок 4.9.5)
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Логування самої моделі як артефакту
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Завершено запуск: max_depth={depth}, accuracy={acc:.4f}")

        # Створи папку для моделей, якщо її немає
os.makedirs('models', exist_ok=True)

# Після циклу навчання збережи останню (або найкращу) модель локально
joblib.dump(model, 'models/model.pkl')
print("Модель успішно збережена в models/model.pkl")