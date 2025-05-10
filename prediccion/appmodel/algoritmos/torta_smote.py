import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Cargar dataset que ya tiene SMOTE aplicado
df_smote = pd.read_csv("prediccion/appmodel/algoritmos/diabetes_prediction_dataset.csv")

# Conteo de clases
clases_smote = df_smote['diabetes'].value_counts()
etiquetas = ['No Diabetes (0)', 'Diabetes (1)']
colores = ['#66b3ff', '#ff9999']

# Gráfico de torta
plt.figure(figsize=(5, 5))
plt.pie(clases_smote, labels=etiquetas, autopct='%1.1f%%', startangle=140, colors=colores)
plt.title("Distribución de clases CON SMOTE")
plt.tight_layout()
plt.show()
