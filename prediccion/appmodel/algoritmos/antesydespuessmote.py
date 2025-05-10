import pandas as pd
import matplotlib.pyplot as plt

# Cargar datasets
df_antes = pd.read_csv("prediccion/appmodel/algoritmos/antes_diabetes_prediction_dataset.csv")
df_despues = pd.read_csv("prediccion/appmodel/algoritmos/diabetes_prediction_dataset.csv")

# Conteo
val_antes = df_antes['diabetes'].value_counts()
val_despues = df_despues['diabetes'].value_counts()

# Etiquetas y colores
labels = ['No Diabetes (0)', 'Diabetes (1)']
colors = ['#66b3ff', '#ff9999']

# Crear figura
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Distribución de Clases Antes y Después de Aplicar SMOTE", fontsize=16, weight='bold')

# --- ANTES de SMOTE ---
# Texto
texto_antes = (
    f"📊 Datos ANTES de SMOTE\n\n"
    f"🩸 No Diabetes (0): {val_antes[0]:,} registros\n"
    f"🧪 Diabetes (1): {val_antes[1]:,} registros"
)
axes[0, 0].axis('off')
axes[0, 0].text(0.02, 0.5, texto_antes, fontsize=13, va='center')

# Gráfico torta
axes[0, 1].pie(val_antes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
axes[0, 1].set_title("Porcentaje antes de SMOTE", fontsize=12)

# --- DESPUÉS de SMOTE ---
# Texto
texto_despues = (
    f"📊 Datos DESPUÉS de SMOTE\n\n"
    f"🩸 No Diabetes (0): {val_despues[0]:,} registros\n"
    f"🧪 Diabetes (1): {val_despues[1]:,} registros"
)
axes[1, 0].axis('off')
axes[1, 0].text(0.02, 0.5, texto_despues, fontsize=13, va='center')

# Gráfico torta
axes[1, 1].pie(val_despues, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
axes[1, 1].set_title("Porcentaje después de SMOTE", fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
