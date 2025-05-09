import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener la ruta del directorio base
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'algoritmos', 'diabetes_prediction_dataset.csv')

# Leer el archivo CSV
print("Leyendo el archivo CSV...")
df = pd.read_csv(data_path)
print("Datos cargados correctamente.")

# Convertir la columna 'age' a int
df['age'] = df['age'].astype(int)

# Remapear las categorías en la columna 'smoking_history'
smoking_mapping = {
    'never': 'non-smoker',
    'No Info': 'non-smoker',
    'current': 'current',
    'ever': 'past-smoker',
    'former': 'past-smoker',
    'not current': 'past-smoker'
}
df['smoking_history'] = df['smoking_history'].replace(smoking_mapping)

# One-hot encoding de variables categóricas
df_encoded = pd.get_dummies(df, columns=['gender', 'smoking_history'])

# Convertir columnas booleanas a int
boolean_columns = df_encoded.select_dtypes(include=["bool"]).columns
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)

# Separar las características y la columna objetivo
X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Guardar el escalador
scaler_path = os.path.join(base_dir, 'algoritmos', 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler guardado en: {scaler_path}")

# Entrenar y guardar cada modelo
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Guardar cada modelo
    model_path = os.path.join(base_dir, 'algoritmos', f'{name.lower().replace(" ", "_")}_model.pkl')
    joblib.dump(clf, model_path)
    print(f"Modelo {name} guardado en: {model_path}")


