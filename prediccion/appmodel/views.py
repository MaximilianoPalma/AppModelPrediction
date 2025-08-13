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
import plotly.graph_objects as go
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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import date, timedelta, datetime
import re
import urllib3
import json
from django.core.mail import EmailMessage

@login_required
def base(request):
    return render(request, 'base.html')

def login(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('inicio')
    else:
        form = CustomLoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def index(request):
    try:
        datos_reporte = calcular_datos_reporte()
        graficos = generarGraficos()

        context = {
            'graficos': graficos,
            'promedio_diario': datos_reporte['promedio_diario'],
            'total_semanal': datos_reporte['total_semanal'],
            'casos_totales': datos_reporte['casos_totales'],
            'nuevos_casos_diarios': datos_reporte['nuevos_casos_diarios'],
            'nuevos_casos_semanales': datos_reporte['nuevos_casos_semanales'],
            'test_realizados': datos_reporte['test_realizados'],
            'fallecidos_totales': datos_reporte['fallecidos_totales'],
            'fallecidos_semanales': datos_reporte['fallecidos_semanales'],
        }

        return render(request, 'inicio.html', context)

    except Exception as e:
        print(f"❌ Error al generar gráficos o reporte: {e}")
        return render(request, 'inicio.html', {
            'graficos': [],
            'error': 'Error al generar el reporte o los gráficos.'
        })


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
        observaciones = request.POST['observaciones']

        # Verificar si el RUT ya existe
        if Paciente.objects.filter(rut=rut).exists():
            context = {
                'error_rut': f'Ya existe un paciente registrado con el RUT {rut}',
                'rut': rut,
                'nombre': nombre,
                'apellido': apellido,
                'edad': edad,
                'nacimiento': nacimiento,
                'genero': genero,
                'bmi': bmi,
                'hipertension': hipertension,
                'enfermedad_cardiaca': enfermedad_cardiaca,
                'nivel_hba1c': nivel_hba1c,
                'nivel_glucosa': nivel_glucosa,
                'historial_tabaquismo': historial_tabaquismo,
                'observaciones': observaciones
            }
            return render(request, 'registro.html', context)

        try:
            fecha_formateada = datetime.strptime(nacimiento, '%d-%m-%Y').date()
        except ValueError:
            return HttpResponse("Error en el formato de la fecha. Debe ser DD-MM-YYYY.")

        Paciente.objects.create(
            rut=rut,
            nombre=nombre,
            apellido=apellido,
            edad=edad,
            nacimiento=fecha_formateada,
            genero=genero,
            bmi=bmi,
            hipertension=hipertension,
            enfermedad_cardiaca=enfermedad_cardiaca,
            nivel_hba1c=nivel_hba1c,
            nivel_glucosa=nivel_glucosa,
            historial_tabaquismo=historial_tabaquismo,
            observaciones=observaciones
        )

        return redirect('listado_pacientes')

    return render(request, 'registro.html')

@login_required
def editar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)

    if request.method == 'POST':
        paciente.rut = request.POST.get('rut', paciente.rut)
        paciente.nombre = request.POST.get('pac_nombre', paciente.nombre)
        paciente.apellido = request.POST.get('pac_apellido', paciente.apellido)
        paciente.edad = request.POST.get('pac_edad', paciente.edad)
        paciente.nacimiento = request.POST.get('pac_nacimiento', paciente.nacimiento)
        paciente.genero = request.POST.get('pac_genero', paciente.genero)
        paciente.bmi = request.POST.get('bmi', paciente.bmi)
        paciente.hipertension = request.POST.get('hypertension', paciente.hipertension)
        paciente.enfermedad_cardiaca = request.POST.get('heart_disease', paciente.enfermedad_cardiaca)
        paciente.nivel_hba1c = request.POST.get('hba1c_level', paciente.nivel_hba1c)
        paciente.nivel_glucosa = request.POST.get('blood_glucose_level', paciente.nivel_glucosa)
        paciente.historial_tabaquismo = request.POST.get('smoking_history', paciente.historial_tabaquismo)
        paciente.observaciones = request.POST.get('pac_observaciones', paciente.observaciones)
        
        paciente.save()
        messages.success(request, "Paciente actualizado exitosamente.")
        return redirect('listado_pacientes')
    
    return render(request, 'listado.html', {'paciente': paciente})

@login_required
def eliminar_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)
    
    if request.method == 'POST':
        paciente.delete()
        messages.success(request, "Paciente eliminado exitosamente.")
        return redirect('listado_pacientes') 
    
    return render(request, 'listado.html', {'paciente': paciente})

@login_required
def consulta_paciente(request):
    paciente = None
    if request.method == 'POST':
        rut = request.POST.get('rut')
        # Normalización mínima y no intrusiva: quitar espacios y subir 'k' a 'K' si es DV
        if rut is not None:
            rut = rut.strip()
            if len(rut) >= 1 and rut[-1].lower() == 'k':
                rut = rut[:-1] + 'K'
        # Ajuste mínimo: si el DV es 'k' minúscula, pasarlo a 'K' manteniendo el formato original
        if rut and len(rut) >= 2 and rut[-1] in ('k', 'K'):
            rut = rut[:-1] + rut[-1].upper()
        # Fallback mínimo: si no hay guión, probar con guión antes del DV
        candidato_con_guion = None
        if rut and '-' not in rut and len(rut) >= 2:
            candidato_con_guion = f"{rut[:-1]}-{rut[-1]}"
        try:
            paciente = Paciente.objects.get(rut=rut)
        except Paciente.DoesNotExist:
            # Intento con variante con guión (si aplica)
            if candidato_con_guion:
                try:
                    paciente = Paciente.objects.get(rut=candidato_con_guion)
                except Paciente.DoesNotExist:
                    paciente = None
            else:
                paciente = None
            # Respaldo final: comparar por RUT normalizado contra los registrados
            if paciente is None and rut:
                def _normalizar(s):
                    return re.sub(r"[^0-9kK]", "", s).upper()
                objetivo = _normalizar(rut)
                for p in Paciente.objects.all():
                    if _normalizar(p.rut) == objetivo:
                        paciente = p
                        break

    return render(request, 'consulta.html', {'paciente': paciente})

def formatear_rut(rut):
    rut = str(rut)
    cuerpo, verificador = rut[:-1], rut[-1]
    cuerpo = re.sub(r"\B(?=(\d{3})+(?!\d))", ".", cuerpo)
    return f"{cuerpo}{verificador}"

def reporte_paciente(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)
    
    hoy = date.today()
    nacimiento = paciente.nacimiento
    
    edad_anos = hoy.year - nacimiento.year - ((hoy.month, hoy.day) < (nacimiento.month, nacimiento.day))
    
    if hoy.month >= nacimiento.month:
        edad_meses = hoy.month - nacimiento.month
    else:
        edad_meses = 12 - (nacimiento.month - hoy.month)

    if hoy.day >= nacimiento.day:
        edad_dias = hoy.day - nacimiento.day
    else:
        edad_meses -= 1
        if edad_meses < 0:
            edad_meses = 11
            edad_anos -= 1
        ultimo_dia_mes_anterior = (hoy.replace(day=1) - timedelta(days=1)).day
        edad_dias = ultimo_dia_mes_anterior - nacimiento.day + hoy.day
    
    edad_completa = f"{edad_anos} años, {edad_meses} meses, {edad_dias} días"
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="reporte_paciente_{paciente.rut}.pdf"'
    
    pdf = SimpleDocTemplate(response, pagesize=letter)
    
    estilos = getSampleStyleSheet()
    
    elementos = []

    estilo_titulo = estilos['Title']
    estilo_titulo.textColor = colors.white
    estilo_titulo.backColor = colors.HexColor('#0066cc') 
    estilo_titulo.fontSize = 14
    titulo = Paragraph(f"<b>Reporte Médico</b>", estilo_titulo)
    elementos.append(titulo)
    elementos.append(Spacer(1, 12))

    azul_titulo = colors.HexColor('#0066cc') 
    azul_palido = colors.HexColor('#e6f3ff') 

    rut_formateado = formatear_rut(paciente.rut)

    data_personales = [
        ['Datos Personales', ''],
        ['RUT', rut_formateado],
        ['Nombre', f"{paciente.nombre} {paciente.apellido}"],
        ['Edad', edad_completa],
        ['Género', 'Hombre' if paciente.genero == 1 else 'Mujer' if paciente.genero == 0 else 'Otro']
    ]
    tabla_personales = Table(data_personales, colWidths=['*', '*'])
    tabla_personales.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)), ('VALIGN', (0, 1), (-1, -1), 'TOP'),  
        ('BACKGROUND', (0, 0), (1, 0), azul_titulo),  
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),  
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), azul_palido), 
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black), 
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
    ]))
    elementos.append(tabla_personales)
    elementos.append(Spacer(1, 12))

    data_medicos = [
        ['Datos Médicos', ''],
        ['Índice de Masa Corporal (BMI)', paciente.bmi],
        ['Hipertensión', 'Sí' if paciente.hipertension == 1 else 'No'],
        ['Enfermedad Cardiaca', 'Sí' if paciente.enfermedad_cardiaca == 1 else 'No'],
        ['Nivel de HbA1c', paciente.nivel_hba1c],
        ['Nivel de Glucosa en Sangre', paciente.nivel_glucosa],
        ['Historial de Tabaquismo', 'Nunca' if paciente.historial_tabaquismo == 0 else 'Ex-fumador' if paciente.historial_tabaquismo == 1 else 'Fumador ocasional' if paciente.historial_tabaquismo == 2 else 'Fumador habitual']
    ]
    tabla_medicos = Table(data_medicos, colWidths=['*', '*'])
    tabla_medicos.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (1, 0), azul_titulo),  
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),  
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), azul_palido), 
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black), 
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
    ]))
    elementos.append(tabla_medicos)
    elementos.append(Spacer(1, 12))

    data_observaciones = [
        ['Observaciones Médicas', ''],
        ['Observaciones', paciente.observaciones if paciente.observaciones else 'No hay observaciones registradas']
    ]
    tabla_observaciones = Table(data_observaciones, colWidths=['*', '*'])
    tabla_observaciones.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),  
        ('BACKGROUND', (0, 0), (1, 0), azul_titulo),  
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),  
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), azul_palido),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
    ]))
    elementos.append(tabla_observaciones)
    pdf.build(elementos)
    return response

@login_required
def enviar_reporte(request, paciente_id):
    paciente = get_object_or_404(Paciente, id=paciente_id)

    if request.method == "POST":
        destinatario_email = request.POST.get("email")

        if not destinatario_email:
            return HttpResponse("Debe proporcionar una dirección de correo electrónico.", status=400)

        hoy = date.today()
        nacimiento = paciente.nacimiento

        edad_anos = hoy.year - nacimiento.year - ((hoy.month, hoy.day) < (nacimiento.month, nacimiento.day))

        if hoy.month >= nacimiento.month:
            edad_meses = hoy.month - nacimiento.month
        else:
            edad_meses = 12 - (nacimiento.month - hoy.month)

        if hoy.day >= nacimiento.day:
            edad_dias = hoy.day - nacimiento.day
        else:
            edad_meses -= 1
            if edad_meses < 0:
                edad_meses = 11
                edad_anos -= 1
            ultimo_dia_mes_anterior = (hoy.replace(day=1) - timedelta(days=1)).day
            edad_dias = ultimo_dia_mes_anterior - nacimiento.day + hoy.day

        edad_completa = f"{edad_anos} años, {edad_meses} meses, {edad_dias} días"

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="reporte_paciente_{paciente.rut}.pdf"'
        pdf = SimpleDocTemplate(response, pagesize=letter)
        estilos = getSampleStyleSheet()
        elementos = []

        estilo_titulo = estilos['Title']
        estilo_titulo.textColor = colors.white
        estilo_titulo.backColor = colors.HexColor('#0066cc') 
        estilo_titulo.fontSize = 14
        titulo = Paragraph(f"<b>Reporte Médico</b>", estilo_titulo)
        elementos.append(titulo)
        elementos.append(Spacer(1, 12))

        azul_titulo = colors.HexColor('#0066cc') 
        azul_palido = colors.HexColor('#e6f3ff') 

        rut_formateado = formatear_rut(paciente.rut)

        data_personales = [
            ['Datos Personales', ''],
            ['RUT', rut_formateado],
            ['Nombre', f"{paciente.nombre} {paciente.apellido}"],
            ['Edad', edad_completa],
            ['Género', 'Hombre' if paciente.genero == 1 else 'Mujer' if paciente.genero == 0 else 'Otro']
        ]

        tabla_personales = Table(data_personales, colWidths=['*', '*'])
        tabla_personales.setStyle(TableStyle([
            ('SPAN', (0, 0), (1, 0)), ('VALIGN', (0, 1), (-1, -1), 'TOP'),  
            ('BACKGROUND', (0, 0), (1, 0), azul_titulo),  
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),  
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), azul_palido), 
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black), 
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ]))
        elementos.append(tabla_personales)
        elementos.append(Spacer(1, 12))

        pdf.build(elementos)

        email = EmailMessage(
            subject='Reporte de Paciente',
            body=(
                f'Estimado/a {paciente.nombre} {paciente.apellido},\n\n'
                'Adjunto a este correo se encuentra el reporte médico solicitado.\n\n'
                'Saludos cordiales.'
            ),
            from_email='tu_correo@example.com',
            to=[destinatario_email],
        )
        email.attach(f'reporte_paciente_{paciente.rut}.pdf', response.getvalue(), 'application/pdf')
        email.send()

        return HttpResponse("Correo enviado exitosamente.")

    return HttpResponse("Método no permitido.", status=405)

@login_required
def aplicacion(request):
    def calcular_nivel_riesgo_general(rf, svm, lr):
        promedio = (rf + svm + lr) / 3
        if promedio <= 30:
            return "bajo"
        elif promedio <= 60:
            return "medio"
        else:
            return "alto"

    def parse_fecha_flexible(valor):
        """Acepta 'dd-mm-YYYY' o 'YYYY-mm-dd' y elimina posibles comillas tipográficas."""
        if not valor:
            raise ValueError("La fecha de nacimiento es obligatoria.")
        limpio = valor.strip().strip('"“”')
        from datetime import datetime as _dt
        for fmt in ('%Y-%m-%d', '%d-%m-%Y'):
            try:
                return _dt.strptime(limpio, fmt).date()
            except ValueError:
                continue
        raise ValueError("Formato de fecha inválido. Use DD-MM-YYYY.")

    if request.method == 'POST':
        action = request.POST.get('action')
        try:
            rut = request.POST['rut']
            nombre = request.POST['nombre']
            apellido = request.POST['apellido']
            age = request.POST.get('age')
            nacimiento_input = request.POST['nacimiento']
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

            # Parseo robusto de la fecha para evitar error "invalid date format"
            try:
                nacimiento_date = parse_fecha_flexible(nacimiento_input)
            except ValueError as fe:
                return render(request, 'aplicacion.html', {
                    'error': str(fe),
                    'rut': rut, 'nombre': nombre, 'apellido': apellido,
                    'age': age, 'nacimiento': nacimiento_input,
                    'gender': gender, 'bmi': bmi, 'hypertension': hypertension,
                    'heart_disease': heart_disease, 'hba1c_level': hba1c_level,
                    'blood_glucose_level': blood_glucose_level, 'smoking_history': smoking_history
                })

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

            nivel_riesgo_general = calcular_nivel_riesgo_general(rf_proba, svm_risk[0], logistic_proba)
            promedio_prediccion = (rf_proba + svm_risk[0] + logistic_proba) / 3

            context = {
                # Mantener datos del formulario
                'rut': rut,
                'nombre': nombre,
                'apellido': apellido,
                'age': age,
                'nacimiento': nacimiento_date.strftime('%d-%m-%Y'),
                'gender': gender,
                'bmi': bmi,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'hba1c_level': hba1c_level,
                'blood_glucose_level': blood_glucose_level,
                'smoking_history': smoking_history,
                
                # Resultados de predicción
                'rf_prediction': round(rf_proba, 2),
                'svm_prediction': round(svm_risk[0], 2),
                'logistic_prediction': round(logistic_proba, 2),
                'HbA1c_level': hba1c_level,
                'nivel_riesgo_general': nivel_riesgo_general,
                'promedio_prediccion': round(promedio_prediccion, 2),
                'prediccion_realizada': True  # Flag para mostrar resultados
            }

            if action == 'guardar_analisis':
                Paciente.objects.create(
                    rut=rut,
                    nombre=nombre,
                    apellido=apellido,
                    edad=age,
                    nacimiento=nacimiento_date,
                    genero=gender,
                    bmi=bmi,
                    hipertension=hypertension,
                    enfermedad_cardiaca=heart_disease,
                    nivel_hba1c=hba1c_level,
                    nivel_glucosa=blood_glucose_level,
                    historial_tabaquismo=smoking_history
                )
                context['registro_exitoso'] = True
                
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
        # Ajuste mínimo: quitar espacios y subir 'k' final a 'K' si corresponde
        if rut is not None:
            rut = rut.strip()
            if len(rut) >= 1 and rut[-1].lower() == 'k':
                rut = rut[:-1] + 'K'
        # Fallback mínimo: si no hay guión, probar con guión antes del DV
        candidato_con_guion = None
        if rut and '-' not in rut and len(rut) >= 2:
            candidato_con_guion = f"{rut[:-1]}-{rut[-1]}"
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
            # Intento con variante con guión (si aplica)
            if candidato_con_guion:
                try:
                    paciente = Paciente.objects.get(rut=candidato_con_guion)
                except Paciente.DoesNotExist:
                    paciente = None
            # Respaldo final: comparar por RUT normalizado contra los registrados
            if paciente is None and rut:
                def _normalizar(s):
                    return re.sub(r"[^0-9kK]", "", s).upper()
                objetivo = _normalizar(rut)
                for p in Paciente.objects.all():
                    if _normalizar(p.rut) == objetivo:
                        paciente = p
                        break
            if paciente:
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
    def calcular_nivel_riesgo_general(rf, svm, lr):
        promedio = (rf + svm + lr) / 3
        if promedio <= 30:
            return "bajo"
        elif promedio <= 60:
            return "medio"
        else:
            return "alto"

    if request.method == 'POST':
        try:
            # Obtener el RUT para recuperar los datos del paciente
            rut = request.POST.get('rut')
            paciente = None
            
            if rut:
                try:
                    paciente = Paciente.objects.get(rut=rut)
                except Paciente.DoesNotExist:
                    pass

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

            nivel_riesgo_general = calcular_nivel_riesgo_general(rf_proba, svm_risk[0], logistic_proba)

            context = {
                # Datos del paciente para mantenerlos en pantalla
                'paciente': paciente,
                'nombre': paciente.nombre if paciente else '',
                'apellido': paciente.apellido if paciente else '',
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'hba1c_level': Hba1c_level,
                'blood_glucose_level': blood_glucose_level,
                'smoking_history': smoking_history,
                
                # Resultados de la predicción
                'rf_prediction': round(rf_proba, 2),
                'svm_prediction': round(svm_risk[0], 2),
                'logistic_prediction': round(logistic_proba, 2),
                'HbA1c_level': Hba1c_level,
                'nivel_riesgo_general': nivel_riesgo_general,
                'promedio_prediccion': round((rf_proba + svm_risk[0] + logistic_proba) / 3, 2),
                'prediccion_realizada': True  # Flag para mostrar resultados
            }

            return render(request, 'consulta.html', context)

        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            return render(request, 'consulta.html', {'error': 'Ocurrió un error durante la predicción.'})

    return render(request, 'consulta.html')


@login_required
def descargar_reporte(request):
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((10, 10), "Reporte de Casos de Diabetes en Chile", fill=(0, 0, 0))
    d.text((10, 50), f"Promedio diario de casos: {request.GET.get('promedio_diario', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 70), f"Total semanal de casos: {request.GET.get('total_semanal', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 90), f"Total de exámenes realizados: {request.GET.get('test_realizados', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 110), f"Nuevos casos diarios: {request.GET.get('nuevos_casos_diarios', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 130), f"Nuevos casos semanales: {request.GET.get('nuevos_casos_semanales', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 150), f"Fallecidos reportados esta semana: {request.GET.get('fallecidos_semanales', 'N/A')}", fill=(0, 0, 0))
    d.text((10, 170), f"Fallecidos totales: {request.GET.get('fallecidos_totales', 'N/A')}", fill=(0, 0, 0))

    response = HttpResponse(content_type='image/png')
    response['Content-Disposition'] = 'attachment; filename="reporte_diabetes.png"'
    img.save(response, 'PNG')

    return response

def generar_reporte_vista():
    semana_epidemiologica = "41 semana epidemiológica 2024 (6 al 12 de octubre)"
    promedio_diario_casos = 44
    total_semanal_casos = 306
    test_realizados = 5707
    test_hba1c = 4549 
    test_glucosa = 1158  
    fallecidos_totales = 58017
    fallecidos_semanales = 11
    fallecidos_confirmados = 6
    fallecidos_sospechosos = 5

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

import requests
from bs4 import BeautifulSoup

def obtener_datos_diabetes():
    url = 'https://soched.cl/new/cual-es-la-frecuencia-de-diabetes-en-chile-como-se-si-tengo-diabetes/'

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return "Error al obtener los datos de la fuente."

        soup = BeautifulSoup(response.text, 'html.parser')

        # Recorre todos los párrafos y busca uno que mencione "diabetes" y contenga un %
        for p in soup.find_all('p'):
            texto = p.get_text()
            if 'diabetes' in texto.lower() and '%' in texto:
                print("📌 Párrafo encontrado:", texto)
                return texto

        return "No se encontraron datos actualizados."

    except Exception as e:
        print("❌ Error al hacer scraping:", e)
        return "Error al procesar los datos."


def calcular_datos_reporte():
#    print("🔍 Entrando a calcular_datos_reporte()")

#    try:
        # Intentar obtener la prevalencia real
#        prevalencia_diabetes = obtener_datos_diabetes()
#        print("🌐 Prevalencia obtenida:", prevalencia_diabetes)

#        # Si quieres extraer un número desde el texto prevalencia_diabetes puedes aplicar regex aquí
#        porcentaje_diabetes = 0.123  # Usamos fijo por ahora
#    except Exception as e:
#        print("❌ Error obteniendo datos web:", e)
#        porcentaje_diabetes = 0.123  # Valor por defecto

    try:
        porcentaje_diabetes = 0.123  # Usamos fijo por ahora
        
        poblacion_total = 19492603  # Chile 2023
        incremento_anual = 0.00414  # Estimación

        casos_totales = poblacion_total * porcentaje_diabetes
        nuevos_casos_anuales = poblacion_total * incremento_anual
        nuevos_casos_diarios = nuevos_casos_anuales / 365
        nuevos_casos_semanales = nuevos_casos_diarios * 7

        # Ejemplo de valores reales o estimados
        total_semanal = 306
        promedio_diario = total_semanal / 7
        test_realizados = 5707
        fallecidos_totales = 58017
        fallecidos_semanales = 11

        datos = {
            'casos_totales': int(casos_totales),
            'nuevos_casos_diarios': int(nuevos_casos_diarios),
            'nuevos_casos_semanales': int(nuevos_casos_semanales),
            'promedio_diario': int(promedio_diario),
            'total_semanal': int(total_semanal),
            'test_realizados': test_realizados,
            'fallecidos_totales': fallecidos_totales,
            'fallecidos_semanales': fallecidos_semanales,
        }

        return datos

    except Exception as e:
        print("❌ Error en cálculo de datos:", e)
        # Valores de respaldo
        return {
            'casos_totales': 0,
            'nuevos_casos_diarios': 0,
            'nuevos_casos_semanales': 0,
            'promedio_diario': 0,
            'total_semanal': 0,
            'test_realizados': 0,
            'fallecidos_totales': 0,
            'fallecidos_semanales': 0,
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

from django.contrib.auth.decorators import login_required
from appmodel.Reporteseinformes import generar_graficos  # Importar la función de generación de gráficos
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

def extraerDatosJSON():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url = "https://reportesrem.minsal.cl/?_token=KSVIcB6RieJr6t2NiQ3cUyx38ELjtKXM7faUUrzV&serie=5&rem=117&seccion_id=1417&tipo=3&tipoReload=3&regiones=0&regionesReload=0&servicios=-1&serviciosReload=-1&periodo=2020&mes_inicio=6&mes_final=6"

    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", {"class": "table-bordered"})
    rows = table.find_all("tr")
    data = []

    for row in rows:
        cols = row.find_all("td")
        cols = [col.text.strip() for col in cols]
        if cols:
            data.append(cols)

    output_dir = os.path.join(settings.BASE_DIR, 'static', 'reportes')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'datos.json')
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    return json_path

def generarGraficos():
    json_path = extraerDatosJSON()

    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    df = pd.DataFrame(data)

    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = ["CONCEPTO"] + [f"Columna_{i}" for i in range(1, df.shape[1])]
    df["CONCEPTO"] = df["CONCEPTO"].str.strip()

    diabeticos = df[df["CONCEPTO"] == "DIABETICOS"].iloc[0]
    tabaquismo = df[df["CONCEPTO"] == "TABAQUISMO ≥ 55 AÑOS"].iloc[0]
    enfermedad_renal = df[df["CONCEPTO"].str.contains("ETAPA", na=False)]

    def extract_numeric(series):
        return series[1:15].apply(pd.to_numeric, errors='coerce').fillna(0)

    diabeticos_data = extract_numeric(diabeticos)
    tabaquismo_data = extract_numeric(tabaquismo)
    enfermedad_renal_data = enfermedad_renal.iloc[:, 1:15].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=0)

    grupos_edad = [
        "15 a 19 años", "20 a 24 años", "25 a 29 años", "30 a 34 años",
        "35 a 39 años", "40 a 44 años", "45 a 49 años", "50 a 54 años",
        "55 a 59 años", "60 a 64 años", "65 a 69 años", "70 a 74 años",
        "75 a 79 años", "80 y más años"
    ]

    output_dir = os.path.join(settings.BASE_DIR, 'static')
    os.makedirs(output_dir, exist_ok=True)

    graficos = []

    config = {
        'scrollZoom': False,  # Deshabilitar el zoom con el scroll del mouse
        'displayModeBar': False,  # Ocultar la barra de herramientas
        'doubleClick': 'reset',  # Deshabilitar el zoom con doble clic
        'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'lasso2d']  # Remover herramientas de zoom y selección
    }

    # Gráfico 1 - Distribución de Pacientes Diabéticos por Grupo de Edad
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=grupos_edad, y=diabeticos_data, name='Diabéticos'))
    fig1.update_layout(
        title='Distribución de Pacientes Diabéticos por Grupo de Edad',
        xaxis_title='Grupos de Edad',
        yaxis_title='Cantidad de Pacientes',
        xaxis_tickangle=-45
    )
    grafico_1 = os.path.join(output_dir, 'afico_edad.html')
    fig1.write_html(grafico_1, config=config)
    graficos.append('/static/afico_edad.html')

    # Gráfico 2 - Comparación entre Tabaquismo y Diabetes por Grupo de Edad
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=grupos_edad, y=tabaquismo_data, name='Tabaquismo'))
    fig2.add_trace(go.Bar(x=grupos_edad, y=diabeticos_data, name='Diabéticos'))
    fig2.update_layout(
        title='Comparación entre Tabaquismo y Diabetes por Grupo de Edad',
        xaxis_title='Grupos de Edad',
        yaxis_title='Cantidad de Personas',
        xaxis_tickangle=45,
        barmode='stack'
    )
    grafico_2 = os.path.join(output_dir, 'afico_comparacion.html')
    fig2.write_html(grafico_2, config=config)
    graficos.append('/static/afico_comparacion.html')

    # Gráfico 3 - Relación entre Tabaquismo, Diabetes y Enfermedad Renal Crónica
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=grupos_edad, y=tabaquismo_data, name='Tabaquismo'))
    fig3.add_trace(go.Bar(x=grupos_edad, y=diabeticos_data, name='Diabéticos'))
    fig3.add_trace(go.Bar(x=grupos_edad, y=enfermedad_renal_data, name='Enfermedad Renal Crónica'))
    fig3.update_layout(
        title='Relación entre Tabaquismo, Diabetes y Enfermedad Renal Crónica',
        xaxis_title='Grupos de Edad',
        yaxis_title='Cantidad de Personas',
        xaxis_tickangle=-45,
        barmode='stack'
    )
    grafico_3 = os.path.join(output_dir, 'afico_relacion.html')
    fig3.write_html(grafico_3, config=config)
    graficos.append('/static/afico_relacion.html')

    # Gráfico 4 - Progresión de Enfermedad Renal Crónica en Pacientes Diabéticos
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=grupos_edad, y=enfermedad_renal_data))
    fig4.update_layout(
        title='Progresión de Enfermedad Renal Crónica en Pacientes Diabéticos',
        xaxis_title='Grupos de Edad',
        yaxis_title='Cantidad de Pacientes',
        xaxis_tickangle=45
    )
    grafico_4 = os.path.join(output_dir, 'afico_enfermedad_renal.html')
    fig4.write_html(grafico_4, config=config)
    graficos.append('/static/afico_enfermedad_renal.html')

    return [
        '/static/reportes/grafico_edad.html',
        '/static/reportes/grafico_comparacion.html',
        '/static/reportes/grafico_relacion.html',
        '/static/reportes/grafico_enfermedad_renal.html'
    ]

from django.http import JsonResponse

@login_required
def verificar_rut(request):
    """Vista para verificar si un RUT ya está registrado"""
    if request.method == 'GET':
        rut = request.GET.get('rut', '').strip()
        
        if not rut:
            return JsonResponse({'existe': False, 'rut': rut})
        
        # Función para normalizar RUT (eliminar puntos, guiones, espacios y convertir k a mayúscula)
        def normalizar_rut(rut_input):
            # Eliminar todo excepto números y K/k
            solo_numeros_k = re.sub(r'[^0-9kK]', '', rut_input)
            # Convertir K minúscula a mayúscula
            return solo_numeros_k.upper()
        
        # Función para formatear RUT con puntos y guión
        def formatear_rut(rut_normalizado):
            if len(rut_normalizado) >= 8:
                rut_sin_dv = rut_normalizado[:-1]
                dv = rut_normalizado[-1]
                # Agregar puntos cada 3 dígitos desde el final
                rut_con_puntos = ""
                for i, digit in enumerate(reversed(rut_sin_dv)):
                    if i > 0 and i % 3 == 0:
                        rut_con_puntos = "." + rut_con_puntos
                    rut_con_puntos = digit + rut_con_puntos
                return f"{rut_con_puntos}-{dv}"
            return rut_normalizado
        
        # Normalizar el RUT ingresado
        rut_normalizado = normalizar_rut(rut)
        rut_formateado = formatear_rut(rut_normalizado)
        
        # Buscar en la base de datos
        try:
            # Obtener todos los pacientes y normalizar sus RUTs para comparación
            pacientes = Paciente.objects.all()
            
            existe = False
            rut_encontrado = ""
            
            for paciente in pacientes:
                rut_bd_normalizado = normalizar_rut(paciente.rut)
                if rut_bd_normalizado == rut_normalizado:
                    existe = True
                    rut_encontrado = paciente.rut
                    break
            
            return JsonResponse({
                'existe': existe,
                'rut': rut_encontrado if existe else rut_formateado
            })
        except Exception as e:
            return JsonResponse({'existe': False, 'rut': rut, 'error': str(e)})
    
    return JsonResponse({'existe': False, 'error': 'Método no permitido'})