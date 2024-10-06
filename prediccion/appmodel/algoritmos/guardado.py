
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

# Obtener la ruta del directorio base
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'diabetes_project', 'data', 'diabetes_prediction_dataset.csv')

# Leer el archivo CSV
print("Leyendo el archivo CSV...")
df = pd.read_csv(data_path)
print("Datos cargados correctamente.")

df.head()
# Data Dimension
df.shape
# Data Information
df.info()
#La edad debe ser int. La convertiré.
# Convert age column to int
df['age']=df['age'].astype(int)
df.info()
# Statistics Infromation
df.describe()
# Checking for null values
df.isnull().sum()
# Check for Outliers 
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.ylabel('Values')
plt.title('Diagrama de caja para detectar valores atípicos')
plt.show()
# IQR technique

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate quartiles and IQR for numeric columns
Q1 = numeric_columns.quantile(0.25)
Q3 = numeric_columns.quantile(0.75)
IQR = Q3 - Q1

# Set a threshold for considering a point as an outlier
k = 1.5
outliers = ((numeric_columns < (Q1 - k * IQR)) | (numeric_columns > (Q3 + k * IQR))).any(axis=1)

# Display rows without outliers
df_outliers = df[~outliers]
print(f'The Number of original rows: {df.shape[0]}')
print(f'The Number of original columns: {df.shape[1]}')

print(f'\nThe Number of cleaned rows: {df_outliers.shape[0]}')
print(f'The Number of cleaned columns: {df_outliers.shape[1]}')

print(f'\nThe number of outliers records that dropped is: {df.shape[0] - df_outliers.shape[0]}')
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_outliers)
plt.xticks(rotation=90)
plt.ylabel('Values')
plt.title('Nuevo diagrama de caja después de manejar valores atípicos')
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot for the original DataFrame (df)
sns.boxplot(data=df, ax=axes[0])
axes[0].set(title='Original Data', ylabel='Value')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)  # Rotate x-axis labels

# Boxplot for the DataFrame without outliers (df_outliers)
sns.boxplot(data=df_outliers, ax=axes[1])
axes[1].set(title='Diagrama de caja para el DataFrame sin valores atípicos', ylabel='Value')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)  # Rotate x-axis labels

plt.suptitle('Comparación de valores atípicos', fontsize=16)

plt.show()
#The difference is notable in using the IQR technique to handle outliers.
#**Encoding categorical variables**
df_outliers['smoking_history'].value_counts()
#there are so many similar meanings and values we will categorize them again.
# Define a dictionary to map existing categories to new ones
smoking_mapping = {
    'never': 'non-smoker',
    'No Info': 'non-smoker', # My approach here if there is no info I decided to put him in non-smoker category
    'current': 'current',
    'ever': 'past-smoker',
    'former': 'past-smoker',
    'not current': 'past-smoker'
}

# Replace values in the 'smoking_history' column using the mapping dictionary
df['smoking_history'] = df['smoking_history'].replace(smoking_mapping)
df['smoking_history'].value_counts()
encoding_columns = df.select_dtypes(include=["object"]).columns

# Perform one-hot encoding using pandas.get_dummies
df_encoded = pd.get_dummies(df, columns=encoding_columns)

# Print information about the encoded DataFrame
df_encoded.info()
# Select boolean columns
boolean_columns = df_encoded.select_dtypes(include=["bool"]).columns

# Convert boolean columns to integer type (True -> 1, False -> 0)
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Assuming df is your DataFrame with features and target column
# df = ...

# Step 1: Handle class imbalance using SMOTE
X = df_encoded.drop(columns=['diabetes'])  # Features
y = df_encoded['diabetes']  # Target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to the training data only, with sampling_strategy='auto' to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Train a classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Step 3: Evaluate the classifier
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 3 (continued): Evaluate the classifier
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_encoded is your DataFrame with features and target column
# df_encoded = ...

# Step 1: Handle class imbalance using SMOTE
X = df_encoded.drop(columns=['diabetes'])  # Features
y = df_encoded['diabetes']  # Target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to the training data only, with sampling_strategy='auto' to balance the classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Train classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train_resampled, y_train_resampled)

# Step 3: Evaluate classifiers and plot confusion matrix
plt.figure(figsize=(15, 5))

for i, (name, clf) in enumerate(classifiers.items(), 1):
    plt.subplot(1, 3, i)
    y_pred = clf.predict(X_test)
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {name}')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df_encoded is your DataFrame with features
# df_encoded = ...

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)

# Visualize clusters
plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=clusters, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming df_encoded is your DataFrame with the provided columns
# df_encoded = ...

# Select features for clustering
features = df_encoded[['age', 'heart_disease']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Example: clustering into 3 groups
clusters = kmeans.fit_predict(scaled_features)

# Visualize clusters
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Age (scaled)')
plt.ylabel('Heart Disease (scaled)')
plt.title('K-means Clustering of Age and Heart Disease')
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df_encoded is your DataFrame with the provided columns
# df_encoded = ...

# Select features for clustering
features = df_encoded[['blood_glucose_level', 'heart_disease']]

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Example: clustering into 3 groups
clusters = kmeans.fit_predict(features)

# Visualize clusters
plt.scatter(features['blood_glucose_level'], features['heart_disease'], c=clusters, cmap='viridis')
plt.xlabel('blood glucose level')
plt.ylabel('Heart Disease')
plt.title('K-means Clustering of blood glucose level and Heart Disease')
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming heart_disease is your DataFrame with the provided columns
# heart_disease = ...

# Set seed value
seed_val = 10

# Select features for clustering
features = df_encoded[['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                          'blood_glucose_level', 'diabetes', 'gender_Female', 'gender_Male',
                          'gender_Other', 'smoking_history_current', 'smoking_history_non-smoker',
                          'smoking_history_past-smoker']]

# Set seed
import numpy as np
np.random.seed(seed_val)

# Select a number of clusters
k = 5

# Run the k-means algorithms
first_clust = KMeans(n_clusters=k, n_init=1, random_state=seed_val).fit(features)

# How many patients are in each group?
print(first_clust.labels_)
print(pd.Series(first_clust.labels_).value_counts())

# Set the seed
seed_val = 38

# Set seed
np.random.seed(seed_val)

# Run the k-means algorithms
second_clust = KMeans(n_clusters=k, n_init=1, random_state=seed_val).fit(features)

# How many patients are in each group?
print(second_clust.labels_)
print(pd.Series(second_clust.labels_).value_counts())

# Adding cluster assignments to the data
df_encoded['first_clust'] = first_clust.labels_
df_encoded['second_clust'] = second_clust.labels_

# Visualizing the clusters
plt.scatter(df_encoded['age'], df_encoded['diabetes'], c=df_encoded['first_clust'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.title('First Clustering Algorithm')
plt.colorbar(label='Cluster')
plt.show()

plt.scatter(df_encoded['age'], df_encoded['diabetes'], c=df_encoded['second_clust'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Heart Disease')
plt.title('Second Clustering Algorithm')
plt.colorbar(label='Cluster')
plt.show()

#the limitation here is visualization not very useful in this case
#Let's choose the application of **Early Disease Detection** and build a predictive model to identify individuals at risk of developing certain health conditions in the future. We'll focus on predicting the likelihood of developing diabetes based on the available features in the dataset.


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target (y)
X = df_encoded.drop(columns=['diabetes'])  # Features
y = df_encoded['diabetes']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

#**Brief Explanation:**

#Early disease detection is crucial for timely interventions and preventive measures to reduce the risk of disease progression. In this application, we built a predictive model to identify individuals at risk of developing diabetes based on their health and demographic features.

#We used a Random Forest classifier, which is an ensemble learning method known for its robustness and ability to handle complex datasets. The model was trained on a dataset containing features such as age, hypertension, BMI, HbA1c level, blood glucose level, gender, and smoking history.

#The dataset was split into training and testing sets to evaluate the model's performance. We standardized the features to ensure they have a mean of 0 and a variance of 1, which is important for many machine learning algorithms.

#The model's performance was evaluated using accuracy and the classification report, which includes precision, recall, F1-score, and support for each class.

#**Limitations:**

#Data Quality: The predictive model's performance heavily relies on the quality and representativeness of the data. If the dataset contains errors, missing values, or biases, it can affect the model's accuracy and reliability.

#Feature Selection: The choice of features included in the model can significantly impact its predictive performance. It's essential to select relevant features that have a strong association with the target variable (diabetes) and exclude irrelevant or redundant features.

#Imbalanced Data: If the dataset is imbalanced, meaning one class (e.g., individuals with diabetes) is significantly more prevalent than the other class, it can lead to biased model predictions. Techniques such as oversampling, undersampling, or using class weights can be employed to address imbalanced data.

#Model Interpretability: Random Forest models, while powerful, are often considered as black-box models, meaning they lack interpretability compared to simpler models like logistic regression. Understanding how the model makes predictions and interpreting its results can be challenging.

#Generalization: The model's performance on unseen data (i.e., data not used during training) may vary, and it's essential to assess its generalization capability on different datasets to ensure its reliability in real-world scenarios.




#**Another model**
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target (y)
X = df_encoded.drop(columns=['diabetes'])  # Features
y = df_encoded['diabetes']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))
