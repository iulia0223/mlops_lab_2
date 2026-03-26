import os
import json
import pytest

# 1. Перевірка наявності артефактів
def test_artifacts_exist():
    # Виводимо список файлів у лог GitHub, якщо тест впаде
    print(f"Поточна директорія: {os.getcwd()}")
    print(f"Файли в корені: {os.listdir('.')}")
    
    assert os.path.exists("models/model.pkl"), "Файл моделі 'models/model.pkl' не знайдено!"
    assert os.path.exists("metrics.json"), "Файл 'metrics.json' не знайдено в корені проєкту!"
    assert os.path.exists("confusion_matrix.png"), "Графік 'confusion_matrix.png' не знайдено!"

# 2. Quality Gate: Перевірка якості моделі
def test_quality_gate_f1():
    # Читаємо поріг з перемінної оточення або ставимо 0.70
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    
    if not os.path.exists("metrics.json"):
        pytest.fail("Неможливо перевірити Quality Gate: metrics.json відсутній")
    
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    f1 = float(metrics["f1"])
    
    assert f1 >= threshold, f"Quality Gate не пройдено: F1 score {f1:.4f} нижче порогу {threshold}"