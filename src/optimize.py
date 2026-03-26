import pandas as pd
import numpy as np
import optuna
import mlflow
import hydra
import joblib
import os
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    # 1. Налаштування MLflow [cite: 147]
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # 2. Завантаження та передобробка даних [cite: 519]
    if not os.path.exists(cfg.data.processed_path):
        print(f"Error: File not found at {cfg.data.processed_path}")
        return

    train = pd.read_csv(cfg.data.processed_path)
    
    # Видаляємо текстові колонки (Date, Location тощо), які викликали помилку [cite: 10, 513]
    X = train.select_dtypes(include=[np.number])
    
    # Обробка цільової змінної RainTomorrow
    if "RainTomorrow" in train.columns:
        # Перетворюємо Yes/No в 1/0, якщо це ще не зроблено
        y = train["RainTomorrow"].map({'No': 0, 'Yes': 1})
        # Видаляємо цільову змінну з ознак, якщо вона там є
        if "RainTomorrow" in X.columns:
            X = X.drop("RainTomorrow", axis=1)
    else:
        print("Error: Target column 'RainTomorrow' not found!")
        return

    # Заповнюємо пропуски (NaN), бо RandomForest не працює з ними [cite: 519]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    # 3. Визначення Objective Function для Optuna [cite: 11, 53, 137]
    def objective(trial: optuna.Trial):
        # Пропонуємо гіперпараметри для цієї спроби [cite: 20, 141]
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 
                                              cfg.hpo.random_forest.n_estimators.low, 
                                              cfg.hpo.random_forest.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", 
                                           cfg.hpo.random_forest.max_depth.low, 
                                           cfg.hpo.random_forest.max_depth.high),
            "random_state": cfg.seed,
            "n_jobs": -1
        }
        
        # Логуємо кожну спробу як Child Run [cite: 23, 64, 143]
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            model = RandomForestClassifier(**params)
            model.fit(X, y)
            
            # Оцінка якості (для простоти на тому ж наборі) [cite: 11, 269]
            preds = model.predict(X)
            score = f1_score(y, preds)
            
            # Логування параметрів та метрик trial [cite: 157-160]
            mlflow.log_params(params)
            mlflow.log_metric(cfg.hpo.metric, score)
            mlflow.set_tag("trial_number", trial.number)
            
            return score

    # 4. Запуск процесу оптимізації (Parent Run) [cite: 63, 144, 386]
    with mlflow.start_run(run_name="HPO_Study_Parent") as parent_run:
        # Фіксуємо seed для відтворюваності
        if cfg.hpo.get("sampler", "tpe").lower() == "random":
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        study.optimize(objective, n_trials=cfg.hpo.n_trials) # Запуск 20 спроб [cite: 22, 150]
        
        # Логування найкращих результатів у Parent Run [cite: 161-163]
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"best_{cfg.hpo.metric}", study.best_value)
        
        # Збереження найкращої моделі [cite: 24, 156, 406]
        best_model = RandomForestClassifier(**study.best_params, random_state=cfg.seed)
        best_model.fit(X, y)
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_model, model_path)
        
        # Логування артефакту моделі [cite: 156, 165]
        mlflow.log_artifact(model_path)
        
        print(f"Optimization finished! Best score: {study.best_value}")
        print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main()