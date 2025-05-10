import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === 1. CARGAR DATOS CON SMOTE APLICADO ===
df = pd.read_csv("prediccion/appmodel/algoritmos/diabetes_prediction_dataset.csv")
print("✅ Dataset balanceado (SMOTE) cargado correctamente.")

# Separar características y etiqueta
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Escalar variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# === 2. MOSTRAR DISTRIBUCIÓN DE CLASES ===
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribución de clases (dataset con SMOTE)")
plt.xlabel("Diabetes (0 = No, 1 = Sí)")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

# === 3. ENTRENAR Y EVALUAR MODELOS ===
modelos = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500)
}

for nombre, modelo in modelos.items():
    print(f"\n🔍 Modelo: {nombre}")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Reporte en texto
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    # Matriz de Confusión Gráfica
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                  display_labels=modelo.classes_)
    disp.plot(cmap=plt.cm.Greens)
    plt.title(f"Matriz de Confusión - {nombre} (dataset con SMOTE)")
    plt.tight_layout()
    plt.show()
