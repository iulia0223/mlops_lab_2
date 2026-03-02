import pandas as pd
import os

# Шлях до завантаженого датасету [cite: 80, 254]
data_path = os.path.join('data', 'raw', 'weatherAUS.csv')
df = pd.read_csv(data_path)

# 1. Перевірка структури та пропусків [cite: 86]
print("--- Інформація про колонки ---")
print(df.info()) 

print("\n--- Кількість пропусків (Missing Values) ---")
print(df.isnull().sum()) 

# 2. Статистика цільової змінної [cite: 86]
print("\n--- Розподіл RainTomorrow (Дисбаланс) ---")
print(df['RainTomorrow'].value_counts())