import pandas as pd
import os
import sys

# Отримуємо шляхи з аргументів командного рядка
input_file = sys.argv[1] 
output_dir = sys.argv[2] 

os.makedirs(output_dir, exist_ok=True)

# Завантаження та обробка
df = pd.read_csv(input_file)
df = df.dropna(subset=['RainTomorrow']) 

# Збереження результату
df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
print(f"Data prepared and saved to {output_dir}")