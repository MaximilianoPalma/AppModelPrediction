import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === 1. CARGAR Y PREPROCESAR DATOS ORIGINALES ===
ruta_csv = "prediccion/appmodel/algoritmos/antes_diabetes_prediction_dataset.csv"
df = pd.read_csv(ruta_csv)
print("✅ Datos cargados correctamente.")

# Convertir edad a entero
df['age'] = df['age'].astype(int)

# Normalizar categorías de tabaquismo
df['smoking_history'] = df['smoking_history'].replace({
    'never': 'non-smoker',
    'No Info': 'non-smoker',
    'current': 'current',
    'ever': 'past-smoker',
    'former': 'past-smoker',
    'not current': 'past-smoker'
})

# One-hot encoding de variables categóricas
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'])

# Separar características y etiqueta
X = df_encoded.drop('diabetes', axis=1)
y = df_encoded['diabetes']

# Escalar variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# === 2. MOSTRAR DISTRIBUCIÓN DE CLASES ===
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribución de clases antes de SMOTE")
plt.xlabel("Diabetes (0 = No, 1 = Sí)")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

# === 3. DEFINIR MODELOS ===
modelos = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500)
}

# === 4. ENTRENAR Y EVALUAR MODELOS ===
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
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.tight_layout()
    plt.show()
