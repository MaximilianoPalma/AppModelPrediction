from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout
from appmodel.forms import CustomLoginForm
from django.contrib.auth.decorators import login_required
from .models import Paciente

from django.conf import settings
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

@login_required
def base(request):
    return render(request, 'base.html')

def login(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('index')
    else:
        form = CustomLoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def index(request):
    pacientes = Paciente.objects.all()
    return render(request, 'index.html', {'pacientes': pacientes})
    

@login_required
def registro_paciente(request):
    if request.method == 'POST':
        rut = request.POST['rut']
        nombre = request.POST['nombre']
        apellido = request.POST['apellido']
        edad = request.POST['age']
        nacimiento = request.POST['nacimiento']
        genero = request.POST['gender']
        bmi = request.POST['bmi']
        hipertension = request.POST['hypertension']
        enfermedad_cardiaca = request.POST['heart_disease']
        nivel_hba1c = request.POST['hba1c_level']
        nivel_glucosa = request.POST['blood_glucose_level']
        historial_tabaquismo = request.POST['smoking_history']
        
        Paciente.objects.create(
            rut=rut,
            nombre=nombre,
            apellido=apellido,
            edad=edad,
            nacimiento=nacimiento,
            genero=genero,
            bmi=bmi,
            hipertension=hipertension,
            enfermedad_cardiaca=enfermedad_cardiaca,
            nivel_hba1c=nivel_hba1c,
            nivel_glucosa=nivel_glucosa,
            historial_tabaquismo=historial_tabaquismo
        )
        
        return redirect('index') 

    return render(request, 'registro.html')


@login_required
def consulta_paciente(request):
    paciente = None
    if request.method == 'POST':
        rut = request.POST.get('rut')
        
        try:
            paciente = Paciente.objects.get(rut=rut)
        except Paciente.DoesNotExist:
            paciente = None

    return render(request, 'consulta.html', {'paciente': paciente})


@login_required
def aplicacion(request):
    return render(request, 'aplicacion.html')

@login_required
def informe(request):
    return render(request, 'informe.html')

@login_required
def soporte(request):
    return render(request, 'soporte.html')

@login_required
def logout(request):
    auth_logout(request)
    return redirect('login')

@login_required
def consulta_paciente(request):
    paciente = None
    if request.method == 'POST':
        rut = request.POST.get('rut')
        try:

            paciente = Paciente.objects.get(rut=rut)

            return render(request, 'consulta.html', {
                'paciente': paciente,
                'nombre': paciente.nombre,
                'apellido': paciente.apellido,
                'age': paciente.edad,
                'nacimiento': paciente.nacimiento,
                'gender': paciente.genero,
                'bmi': paciente.bmi,
                'hypertension': paciente.hipertension,
                'heart_disease': paciente.enfermedad_cardiaca,
                'hba1c_level': paciente.nivel_hba1c,
                'blood_glucose_level': paciente.nivel_glucosa,
                'smoking_history': paciente.historial_tabaquismo,
            })
        except Paciente.DoesNotExist:

            return render(request, 'consulta.html', {'error': 'Paciente no encontrado'})
    return render(request, 'consulta.html')

base_dir = settings.BASE_DIR

rf_model_path = os.path.join(base_dir, 'appmodel\\algoritmos', 'random_forest_model.pkl')
svm_model_path = os.path.join(base_dir, 'appmodel\\algoritmos', 'support_vector_machine_model.pkl')
logistic_model_path = os.path.join(base_dir, 'appmodel\\algoritmos', 'logistic_regression_model.pkl')
scaler_path = os.path.join(base_dir, 'appmodel\\algoritmos', 'scaler.pkl')

rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)
logistic_model = joblib.load(logistic_model_path)
scaler = joblib.load(scaler_path)

columnas_entrenamiento = scaler.feature_names_in_

@login_required
def evaluacion_riesgo(request):
    if request.method == 'POST':
        try:
            age = int(request.POST.get('age'))
            gender = request.POST.get('gender')
            hypertension = int(request.POST.get('hypertension'))
            heart_disease = int(request.POST.get('heart_disease'))
            smoking_history = request.POST.get('smoking_history')
            bmi = float(request.POST.get('bmi'))
            Hba1c_level = float(request.POST.get('HbA1c_level'))
            blood_glucose_level = float(request.POST.get('blood_glucose_level'))

            input_data = {
                'age': [age],
                'bmi': [bmi],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'HbA1c_level': [Hba1c_level],
                'blood_glucose_level': [blood_glucose_level]
            }

            input_data_df = pd.DataFrame(input_data)
            input_data_df['gender'] = gender
            input_data_df['smoking_history'] = smoking_history

            input_data_encoded = pd.get_dummies(input_data_df, columns=['gender', 'smoking_history'])

            for col in columnas_entrenamiento:
                if col not in input_data_encoded.columns:
                    input_data_encoded[col] = 0

            input_data_encoded = input_data_encoded[columnas_entrenamiento]

            input_data_scaled = scaler.transform(input_data_encoded)

            rf_proba = rf_model.predict_proba(input_data_scaled)[0][1] * 100
            svm_proba = svm_model.decision_function(input_data_scaled)
            logistic_proba = logistic_model.predict_proba(input_data_scaled)[0][1] * 100

            svm_risk = (1 / (1 + np.exp(-svm_proba))) * 100

            context = {
                'rf_prediction': round(rf_proba, 2),
                'svm_prediction': round(svm_risk[0], 2),
                'logistic_prediction': round(logistic_proba, 2),
                'blood_glucose_level': blood_glucose_level,
                'HbA1c_level': Hba1c_level,
                'hypertension': hypertension
            }

            return render(request, 'consulta.html', context)

        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            return render(request, 'consulta.html', {'error': 'Ocurrió un error durante la predicción.'})

    return render(request, 'consulta.html')
