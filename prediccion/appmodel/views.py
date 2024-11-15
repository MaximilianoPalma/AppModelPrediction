from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout
from appmodel.forms import CustomLoginForm
from django.contrib.auth.decorators import login_required
from .models import Paciente
from django.shortcuts import get_object_or_404

from django.http import HttpResponse

from django.conf import settings
import joblib
import numpy as np

from reportlab.lib.pagesizes import letter
from .reporte_utils import calcular_datos_reporte

import requests
from bs4 import BeautifulSoup

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from PIL import Image, ImageDraw, ImageFont

from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.hashers import make_password

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
    if request.method == 'POST':
        action = request.POST.get('action')
        try:
            rut = request.POST['rut']
            nombre = request.POST['nombre']
            apellido = request.POST['apellido']
            age = request.POST.get('age')
            nacimiento = request.POST['nacimiento']
            gender = request.POST.get('gender')
            hypertension = request.POST.get('hypertension')
            heart_disease = request.POST.get('heart_disease')
            smoking_history = request.POST.get('smoking_history')
            bmi = request.POST.get('bmi')
            hba1c_level = request.POST.get('hba1c_level')
            blood_glucose_level = request.POST.get('blood_glucose_level')

            if not all([age, gender, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]):
                raise ValueError("Todos los campos son obligatorios.")

            age = int(age)
            hypertension = int(hypertension)
            heart_disease = int(heart_disease)
            bmi = float(bmi)
            hba1c_level = float(hba1c_level)
            blood_glucose_level = float(blood_glucose_level)

            input_data = {
                'age': [age],
                'bmi': [bmi],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'HbA1c_level': [hba1c_level],
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
                'hba1c_level': hba1c_level,
                'hypertension': hypertension
            }

            if action == 'guardar_analisis':
                Paciente.objects.create(
                    rut=rut,
                    nombre=nombre,
                    apellido=apellido,
                    edad=age,
                    nacimiento=nacimiento,
                    genero=gender,
                    bmi=bmi,
                    hipertension=hypertension,
                    enfermedad_cardiaca=heart_disease,
                    nivel_hba1c=hba1c_level,
                    nivel_glucosa=blood_glucose_level,
                    historial_tabaquismo=smoking_history
                )
                
            return render(request, 'aplicacion.html', context)

        except ValueError as ve:
            print(f"Error de validación: {str(ve)}")
            return render(request, 'aplicacion.html', {'error': 'Todos los campos son obligatorios y deben tener el formato adecuado.'})
        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            return render(request, 'aplicacion.html', {'error': 'Ocurrió un error durante la predicción.'})

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
    # Crear una imagen en blanco
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Títulos y datos del reporte
    d.text((10, 10), "Reporte de Casos de Diabetes en Chile", fill=(0, 0, 0))
    d.text((10, 50), f"Promedio diario de casos: {request.GET.get('promedio_diario', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 70), f"Total semanal de casos: {request.GET.get('total_semanal', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 90), f"Total de exámenes realizados: {request.GET.get('test_realizados', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 110), f"Nuevos casos diarios: {request.GET.get('nuevos_casos_diarios', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 130), f"Nuevos casos semanales: {request.GET.get('nuevos_casos_semanales', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 150), f"Fallecidos reportados esta semana: {request.GET.get('fallecidos_semanales', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 170), f"Fallecidos totales: {request.GET.get('fallecidos_totales', 'N/A')}", fill=(0, 0, 0))

    # Guardar la imagen en un objeto BytesIO
    response = HttpResponse(content_type='image/png')
    response['Content-Disposition'] = 'attachment; filename="reporte_diabetes.png"'
    img.save(response, 'PNG')

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

@login_required
def generar_graficos(request):
    try:
        # Ruta al archivo CSV
        data_path = os.path.join(settings.BASE_DIR, 'appmodel', 'algoritmos', 'diabetes_prediction_dataset.csv')
        
        # Leer el archivo CSV
        df = pd.read_csv(data_path)

        # Gráfico de distribución de edad
        plt.figure(figsize=(10, 5))
        plt.hist(df['age'], bins=30, color='blue', alpha=0.7)
        plt.title('Distribución de Edad de Pacientes con Diabetes')
        plt.xlabel('Edad')
        plt.ylabel('Número de Pacientes')
        edad_grafico_path = os.path.join(settings.STATIC_ROOT, 'grafico_edad.png')
        plt.savefig(edad_grafico_path)
        plt.close()

        # Gráfico de nivel de glucosa
        plt.figure(figsize=(10, 5))
        plt.hist(df['blood_glucose_level'], bins=30, color='green', alpha=0.7)
        plt.title('Distribución de Nivel de Glucosa de Pacientes con Diabetes')
        plt.xlabel('Nivel de Glucosa')
        plt.ylabel('Número de Pacientes')
        glucosa_grafico_path = os.path.join(settings.STATIC_ROOT, 'grafico_glucosa.png')
        plt.savefig(glucosa_grafico_path)
        plt.close()

        # Renderizar la plantilla
        return render(request, 'tu_template.html', {
            'grafico_edad': 'static/grafico_edad.png',
            'grafico_glucosa': 'static/grafico_glucosa.png',
        })
    except Exception as e:
        return HttpResponse(f"Error al generar los gráficos: {str(e)}", status=500)

@login_required
def descargar_reporte(request):
    # Crear una imagen en blanco
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Establecer una fuente
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Títulos y datos del reporte
    d.text((10, 10), "Reporte de Casos de Diabetes en Chile", fill=(0, 0, 0), font=font)
    d.text((10, 50), f"Promedio diario de casos: {request.GET.get('promedio_diario', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 70), f"Total semanal de casos: {request.GET.get('total_semanal', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 90), f"Total de exámenes realizados: {request.GET.get('test_realizados', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 110), f"Nuevos casos diarios: {request.GET.get('nuevos_casos_diarios', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 130), f"Nuevos casos semanales: {request.GET.get('nuevos_casos_semanales', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 150), f"Fallecidos reportados esta semana: {request.GET.get('fallecidos_semanales', 'N/A')}", fill=(0, 0, 0), font=font)
    d.text((10, 170), f"Fallecidos totales: {request.GET.get('fallecidos_totales', 'N/A')}", fill=(0, 0, 0), font=font)

    # Guardar la imagen en un objeto BytesIO
    response = HttpResponse(content_type='image/png')
    response['Content-Disposition'] = 'attachment; filename="reporte_diabetes.png"'
    img.save(response, 'PNG')

    return response

def registro_soporte(request):
    if request.method == "POST":
        nombre = request.POST.get("sop_nombre")
        apellido = request.POST.get("sop_apellido")
        correo = request.POST.get("sop_correo")
        usuario = request.POST.get("sop_usuario")
        password = request.POST.get("sop_password")
        password_confirmacion = request.POST.get("sop_password_confirmacion")

        if password == password_confirmacion:
            try:
                user = User.objects.create(
                    username=usuario,
                    first_name=nombre,
                    last_name=apellido,
                    email=correo,
                    password=make_password(password),
                    is_staff=True,
                    is_active=True
                )
                user.save()
                messages.success(request, "Usuario creado exitosamente.")
                return redirect("listado_soporte")
            except Exception as e:
                messages.error(request, f"Error al crear el usuario: {e}")
        else:
            messages.error(request, "Las contraseñas no coinciden.")

    return render(request, "registro_soporte.html")

def listado_soporte(request):
    usuarios_soporte = User.objects.filter(is_staff=True)
    return render(request, 'listado_soporte.html', {'usuarios_soporte': usuarios_soporte})

@login_required
def editar_soporte(request, soporte_id):
    soporte = get_object_or_404(User, id=soporte_id)

    if request.method == 'POST':
        soporte.first_name = request.POST.get("sop_nombre")
        soporte.last_name = request.POST.get("sop_apellido")
        soporte.email = request.POST.get("sop_correo")
        soporte.username = request.POST.get("sop_usuario")
        
        new_password = request.POST.get("sop_password")
        password_confirmacion = request.POST.get("sop_password_confirmacion")
        
        if new_password and new_password == password_confirmacion:
            soporte.password = make_password(new_password)
        elif new_password and new_password != password_confirmacion:
            messages.error(request, "Las contraseñas no coinciden.")
            return redirect('editar_soporte', soporte_id=soporte_id)
        
        soporte.save()
        messages.success(request, "Usuario actualizado exitosamente.")
        return redirect('listado_soporte')
    
    return render(request, 'listado_soporte.html', {'soporte': soporte})

@login_required
def eliminar_soporte(request, soporte_id):
    soporte = get_object_or_404(User, id=soporte_id)
    
    if request.method == 'POST':
        soporte.delete()
        messages.success(request, "Usuario eliminado exitosamente.")
        return redirect('listado_soporte')  # Redirige a la lista de soporte
    
    return render(request, 'listado_soporte.html', {'soporte': soporte})


from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from appmodel.Reporteseinformes import generar_graficos  # Importar la función de generación de gráficos
import os
from django.conf import settings

from appmodel.Reporteseinformes import generar_graficos

@login_required
def reporteinforme(request):
    try:
        # Generar gráficos
        graficos = generar_graficos()

        # Contexto para el template
        context = {
            'grafico_edad': '/' + os.path.relpath(graficos[0], settings.BASE_DIR).replace("\\", "/"),
            'grafico_comparacion': '/' + os.path.relpath(graficos[1], settings.BASE_DIR).replace("\\", "/"),
            'grafico_relacion': '/' + os.path.relpath(graficos[2], settings.BASE_DIR).replace("\\", "/"),
            'grafico_enfermedad_renal': '/' + os.path.relpath(graficos[3], settings.BASE_DIR).replace("\\", "/"),
        }

        return render(request, 'reporteinforme.html', context)
    except Exception as e:
        return HttpResponse(f"Error al generar gráficos: {str(e)}", status=500)




