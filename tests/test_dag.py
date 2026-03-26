from airflow.models import DagBag

def test_dag_import_errors():
    """Перевірка, чи завантажуються DAG-и без помилок"""
    dag_bag = DagBag(dag_folder='airflow/dags/', include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"