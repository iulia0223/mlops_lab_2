import os
import json
import pytest

# 1. Перевірка наявності артефактів
def test_artifacts_exist():
    assert os.path.exists("models/model.pkl"), "Файл моделі не знайдено!"
    assert os.path.exists("metrics.json"), "Файл метрик metrics.json не знайдено!"
    assert os.path.exists("confusion_matrix.png"), "Графік матриці помилок не знайдено!"

# 2. Quality Gate: Перевірка якості моделі
def test_quality_gate_f1():
    # Читаємо поріг з перемінної оточення або ставимо 0.70 за замовчуванням
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    f1 = float(metrics["f1"])
    
    # Якщо F1 менше порогу, тест завалиться і CI/CD зупиниться
    assert f1 >= threshold, f"Quality Gate failed: F1 score {f1:.4f} is below threshold {threshold}"