import json
import os
from datetime import datetime, timedelta

import joblib
import mlflow
import mlflow.sklearn
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 3, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def evaluate_model(**_kwargs):
    """Зчитує metrics.json та повертає метрики в XCom."""
    metrics_path = "/opt/airflow/metrics.json"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json не знайдено за шляхом {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


def check_accuracy(**kwargs):
    """Гілкування: якщо accuracy вище порогу — реєструємо модель."""
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")
    accuracy = float(metrics.get("accuracy", 0.0))
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))

    if accuracy >= threshold:
        return "register_model"
    return "stop_pipeline"


def register_model(**_kwargs):
    """Реєстрація моделі у MLflow Model Registry зі стадією Staging."""
    model_path = "/opt/airflow/models/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не знайдено за шляхом {model_path}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Rain_Australia_CT")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "rain_australia_model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model = joblib.load(model_path)

    with mlflow.start_run(run_name="ct_register_from_airflow") as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=result.name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=False,
    )


with DAG(
    "ml_training_pipeline_v1",
    default_args=default_args,
    description="Пайплайн для автоматизації Continuous Training (CT)",
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "lab5"],
) as dag:
    check_data = BashOperator(
        task_id="check_raw_data_exists",
        bash_command="test -f /opt/airflow/data/raw/weatherAUS.csv",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            "cd /opt/airflow && "
            "python src/prepare.py data/raw/weatherAUS.csv data/prepared"
        ),
    )

    train_model = BashOperator(
        task_id="train_ml_model",
        bash_command="cd /opt/airflow && python src/train.py",
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch = BranchPythonOperator(
        task_id="accuracy_branch",
        python_callable=check_accuracy,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    stop = EmptyOperator(task_id="stop_pipeline")

    check_data >> prepare_data >> train_model >> evaluate >> branch
    branch >> register
    branch >> stop