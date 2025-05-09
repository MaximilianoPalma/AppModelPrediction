import pandas as pd
import os

# Ruta completa al archivo CSV
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'algoritmos', 'diabetes_prediction_dataset.csv')

# Leer el archivo CSV
df = pd.read_csv(data_path)

# Ver cuántas instancias hay de cada clase (diabetes = 1, no diabetes = 0)
print(df['diabetes'].value_counts())

# También puedes calcular el porcentaje de cada clase
print(df['diabetes'].value_counts(normalize=True) * 100)
