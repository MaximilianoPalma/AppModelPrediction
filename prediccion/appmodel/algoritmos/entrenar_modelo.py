import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# Obtener la ruta del directorio base
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'algoritmos', 'diabetes_prediction_dataset.csv')

# Leer el archivo CSV
print("Leyendo el archivo CSV...")
df = pd.read_csv(data_path)
print("Datos cargados correctamente.")
print("Columnas en el dataset:", df.columns.tolist())

# Convertir la columna 'age' a int si existe
if 'age' in df.columns:
    df['age'] = df['age'].astype(int)

# Remapear categorías en 'smoking_history' si existe
if 'smoking_history' in df.columns:
    smoking_mapping = {
        'never': 'non-smoker',
        'No Info': 'non-smoker',
        'current': 'current',
        'ever': 'past-smoker',
        'former': 'past-smoker',
        'not current': 'past-smoker'
    }
    df['smoking_history'] = df['smoking_history'].replace(smoking_mapping)
else:
    print("Columna 'smoking_history' no encontrada, se omite el remapeo.")

# One-hot encoding de variables categóricas
categorical_cols = []
if 'gender' in df.columns:
    categorical_cols.append('gender')
if 'smoking_history' in df.columns:
    categorical_cols.append('smoking_history')

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Convertir columnas booleanas a int
boolean_columns = df_encoded.select_dtypes(include=["bool"]).columns
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)

# Separar características y etiqueta
X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Aplicar SMOTE para balancear clases
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Guardar scaler
scaler_path = os.path.join(base_dir, 'algoritmos', 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler guardado en: {scaler_path}")

# Entrenar modelos
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train_resampled, y_train_resampled)
    model_path = os.path.join(
        base_dir, 'algoritmos', f'{name.lower().replace(" ", "_")}_model.pkl'
    )
    joblib.dump(clf, model_path)
    print(f"Modelo {name} guardado en: {model_path}")

# Análisis del Random Forest guardado (si quieres)
rf_model_path = os.path.join(base_dir, 'algoritmos', 'random_forest_model.pkl')
if os.path.exists(rf_model_path):
    best_rf = joblib.load(rf_model_path)
    print("\n--- Hiperparámetros Random Forest ---")
    print("Número de árboles:", best_rf.n_estimators)
    print("Profundidad máxima:", best_rf.max_depth)
    print("Mínimo samples split:", best_rf.min_samples_split)
    print("Mínimo samples leaf:", best_rf.min_samples_leaf)
    print("Criterio de división:", best_rf.criterion)

    # Importancia de características
    feature_names = X.columns
    importances = best_rf.feature_importances_
    df_importance = pd.DataFrame({'Característica': feature_names, 'Importancia': importances})
    df_importance = df_importance.sort_values(by='Importancia', ascending=False)
    print("\nImportancia de características:\n", df_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(df_importance['Característica'], df_importance['Importancia'], color='skyblue')
    plt.xlabel('Importancia')
    plt.title('Importancia de características - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("Modelo Random Forest no encontrado para análisis.")
