# --- Етап 1: Builder (Встановлення залежностей) ---
FROM python:3.10-slim as builder

WORKDIR /app

# Копіюємо файл залежностей
COPY requirements.txt .

# Встановлюємо залежності в окрему директорію для легкого копіювання
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Етап 2: Final (Фінальний легкий образ) ---
FROM python:3.10-slim

WORKDIR /app

# Копіюємо встановлені бібліотеки з етапу builder
COPY --from=builder /install /usr/local

# Копіюємо код ML-проєкту та дані
COPY src/ /app/src/
COPY data/ /app/data/
COPY requirements.txt /app/

# Вказуємо точку входу (наприклад, для тренування моделі)
CMD ["python", "src/train.py"]