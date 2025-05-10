import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset original (sin SMOTE)
df = pd.read_csv("prediccion/appmodel/algoritmos/antes_diabetes_prediction_dataset.csv")

# Conteo de clases
clases = df['diabetes'].value_counts()
etiquetas = ['No Diabetes (0)', 'Diabetes (1)']
colores = ['#66b3ff', '#ff9999']

# Gráfico de torta
plt.figure(figsize=(5, 5))
plt.pie(clases, labels=etiquetas, autopct='%1.1f%%', startangle=140, colors=colores)
plt.title("Distribución de clases SIN SMOTE")
plt.tight_layout()
plt.show()
