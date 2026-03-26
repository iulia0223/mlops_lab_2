import json
import os
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.filesystem import FileSensor
import mlflow
import mlflow.sklearn

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML Training Pipeline with DVC and MLflow',
    schedule='@daily',
    catchup=False,
)

# 1. Sensor / Check
check_dvc_file = FileSensor(
    task_id='check_dvc_config',
    filepath='/opt/airflow/project/dvc.yaml',
    fs_conn_id='fs_default',
    poke_interval=10,
    timeout=60,
    dag=dag,
)

# 2. Data Preparation
prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='cd /opt/airflow/project && dvc repro',
    dag=dag,
)

# 3. Model Training
train_model = BashOperator(
    task_id='train_model',
    bash_command='cd /opt/airflow/project && python src/train.py',
    dag=dag,
)

# Функція для Task 4a: Зчитування метрик
def push_metrics_to_xcom(**kwargs):
    metrics_path = '/opt/airflow/project/metrics.json'
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("metrics.json не знайдено.")
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    print(f"Зчитані метрики: {metrics}")
    return metrics

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=push_metrics_to_xcom,
    dag=dag,
)

# Функція для Task 4b: Branching (за методичними вказівками)
def check_accuracy(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    if metrics and metrics.get('accuracy', 0) > 0.85:
        return 'register_model'
    return 'stop_pipeline'

branching = BranchPythonOperator(
    task_id='branching',
    python_callable=check_accuracy,
    dag=dag,
)

# 5. Model Registration
def register_in_mlflow(**kwargs):
    print("Реєстрація моделі в MLflow Staging...")
    mlflow.set_tracking_uri("file:///opt/airflow/project/mlruns")
    mlflow.set_experiment("Production_Models")
    
    import joblib
    # Завантажуємо навчену збережену модель
    model = joblib.load('/opt/airflow/project/models/model.pkl')
    
    with mlflow.start_run() as run:
        # Логуємо артефакти як звичайні файли
        mlflow.log_artifact('/opt/airflow/project/metrics.json')
        mlflow.log_artifact('/opt/airflow/project/confusion_matrix.png')
        
        # Логуємо модель саме як MLflow Model (щоб реєстр працював)
        mlflow.sklearn.log_model(model, "rain_model")
        
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/rain_model"
        
        result = mlflow.register_model(model_uri, "RandomForest_Weather")
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="RandomForest_Weather",
            version=result.version,
            stage="Staging"
        )
    print("Модель успішно зареєстрована у Staging.")

register_model = PythonOperator(
    task_id='register_model',
    python_callable=register_in_mlflow,
    dag=dag,
)

stop_pipeline = BashOperator(
    task_id='stop_pipeline',
    bash_command='echo "Якість моделі низька (Accuracy <= 0.85). Пайплайн завершено без реєстрації."',
    dag=dag,
)

# Задаємо послідовність
check_dvc_file >> prepare_data >> train_model >> evaluate_model >> branching
branching >> [register_model, stop_pipeline]
