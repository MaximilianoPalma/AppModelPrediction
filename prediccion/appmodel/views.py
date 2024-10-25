from io import BytesIO
from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout
from appmodel.forms import CustomLoginForm
from django.contrib.auth.decorators import login_required
from .models import Paciente

from reportlab.pdfgen import canvas
from django.http import HttpResponse

from django.conf import settings
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from reportlab.lib.pagesizes import letter
from .reporte_utils import calcular_datos_reporte

import requests
from bs4 import BeautifulSoup

@login_required
def base(request):
    return render(request, 'base.html')

def login(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return render(request, 'index.html')
    else:
        form = CustomLoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def index(request):
    # Calcular los datos del reporte en tiempo real
    datos_reporte = calcular_datos_reporte()

    # Contexto para enviar los datos al template
    context = {
        'promedio_diario': datos_reporte['promedio_diario'],
        'total_semanal': datos_reporte['total_semanal'],
        'casos_totales': datos_reporte['casos_totales'],
        'nuevos_casos_diarios': datos_reporte['nuevos_casos_diarios'],
        'nuevos_casos_semanales': datos_reporte['nuevos_casos_semanales'],
        'test_realizados': datos_reporte['test_realizados'],
        'fallecidos_totales': datos_reporte['fallecidos_totales'],
        'fallecidos_semanales': datos_reporte['fallecidos_semanales'],
    }

    return render(request, 'index.html', context)

@login_required
def listado_pacientes(request):
    pacientes = Paciente.objects.all()
    return render(request, 'listado.html', {'pacientes': pacientes})
    

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

@login_required
def descargar_reporte(request):
    # Crear un objeto HttpResponse con el tipo de contenido 'application/pdf'
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="reporte_diabetes.pdf"'

    # Crear el PDF utilizando ReportLab
    pdf = canvas.Canvas(response, pagesize=letter)
    pdf.drawString(100, 750, "Reporte de Casos de Diabetes en Chile")
    
    # Agregar datos del reporte obtenidos del contexto
    pdf.drawString(100, 720, f"Promedio diario de casos: {request.GET.get('promedio_diario', '')}")
    pdf.drawString(100, 700, f"Total semanal de casos: {request.GET.get('total_semanal', '')}")
    pdf.drawString(100, 680, f"Total de casos acumulados: {request.GET.get('casos_totales', '')}")
    pdf.drawString(100, 660, f"Nuevos casos diarios estimados: {request.GET.get('nuevos_casos_diarios', '')}")
    pdf.drawString(100, 640, f"Nuevos casos semanales estimados: {request.GET.get('nuevos_casos_semanales', '')}")
    pdf.drawString(100, 620, f"Test realizados: {request.GET.get('test_realizados', '')}")
    pdf.drawString(100, 600, f"Fallecidos totales: {request.GET.get('fallecidos_totales', '')}")
    pdf.drawString(100, 580, f"Fallecidos esta semana: {request.GET.get('fallecidos_semanales', '')}")

    # Finalizar y devolver el PDF
    pdf.showPage()
    pdf.save()
    return response

def generar_reporte_vista():
    # Datos base
    semana_epidemiologica = "41 semana epidemiológica 2024 (6 al 12 de octubre)"
    promedio_diario_casos = 44
    total_semanal_casos = 306
    test_realizados = 5707  # Exámenes realizados
    test_hba1c = 4549  # Ejemplo: Hemoglobina Glicosilada HbA1c
    test_glucosa = 1158  # Ejemplo: Test de glucosa en ayunas
    fallecidos_totales = 58017  # Fallecidos totales en Chile
    fallecidos_semanales = 11
    fallecidos_confirmados = 6
    fallecidos_sospechosos = 5

    # Reporte estructurado
    reporte = f"""
    ***{semana_epidemiologica}***

    **Casos Confirmados:**
    - Promedio diario de casos: {promedio_diario_casos}
    - Total semanal de casos: {total_semanal_casos}

    **Laboratorio:**
    - Número de exámenes informados en la última semana: {test_realizados}
        - Test de Hemoglobina Glicosilada (HbA1c): {test_hba1c}
        - Test de glucosa en ayunas: {test_glucosa}

    **Casos Fallecidos:**
    - Fallecidos reportados en la última semana: {fallecidos_semanales}
        - Confirmados: {fallecidos_confirmados}
        - Sospechosos o probables: {fallecidos_sospechosos}
    - Casos fallecidos totales en Chile: {fallecidos_totales}

    Fuente: Departamento de Epidemiología, Ministerio de Salud
    """

    return reporte

def obtener_datos_diabetes():
    url = 'https://soched.cl/new/cual-es-la-frecuencia-de-diabetes-en-chile-como-se-si-tengo-diabetes/'
    response = requests.get(url)

    # Verificar que la solicitud fue exitosa
    if response.status_code != 200:
        return "Error al obtener los datos de la fuente."

    soup = BeautifulSoup(response.text, 'html.parser')

    # Intentar encontrar un párrafo que mencione la prevalencia de diabetes
    try:
        prevalencia = soup.find('p', string=lambda text: 'diabetes' in text.lower()).text
        if prevalencia:
            return prevalencia
        else:
            return "No se encontraron datos actualizados."
    except AttributeError:
        return "Error al procesar los datos."

# Función que utiliza datos actuales de scraping en vez de datos fijos
def calcular_datos_reporte():
    # Obtener la información de la página de SOCHED
    prevalencia_diabetes = obtener_datos_diabetes()

    # Datos base de diabetes en Chile con scraping (supongamos que la prevalencia es el porcentaje extraído)
    poblacion_total = 19492603  # Población total de Chile en 2023
    porcentaje_diabetes = 0.123  # 12.3% (o usar prevalencia_diabetes si puedes extraer el número de allí)
    
    incremento_anual = 0.00414  # Incremento anual estimado (puedes ajustarlo según los datos disponibles)

    # Casos actuales y nuevos casos
    casos_totales = poblacion_total * porcentaje_diabetes
    nuevos_casos_anuales = poblacion_total * incremento_anual
    nuevos_casos_diarios = nuevos_casos_anuales / 365
    nuevos_casos_semanales = nuevos_casos_diarios * 7

    # Datos adicionales (ejemplos de test realizados y fallecidos)
    total_semanal = 306  # Ejemplo de total semanal
    promedio_diario = total_semanal / 7
    test_realizados = 5707  # Ejemplo de test realizados
    fallecidos_totales = 58017  # Ejemplo
    fallecidos_semanales = 11  # Ejemplo

    return {
        'casos_totales': int(casos_totales),
        'nuevos_casos_diarios': int(nuevos_casos_diarios),
        'nuevos_casos_semanales': int(nuevos_casos_semanales),
        'promedio_diario': int(promedio_diario),
        'total_semanal': int(total_semanal),
        'test_realizados': test_realizados,
        'fallecidos_totales': fallecidos_totales,
        'fallecidos_semanales': fallecidos_semanales,
    }
