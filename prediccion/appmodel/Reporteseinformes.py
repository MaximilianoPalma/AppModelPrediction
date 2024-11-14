import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import urllib3
from django.conf import settings

def generar_graficos():
    # Ignorar advertencias de SSL
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # URL del sitio web
    url = "https://reportesrem.minsal.cl/?_token=KSVIcB6RieJr6t2NiQ3cUyx38ELjtKXM7faUUrzV&serie=5&rem=117&seccion_id=1417&tipo=3&tipoReload=3&regiones=0&regionesReload=0&servicios=-1&serviciosReload=-1&periodo=2020&mes_inicio=6&mes_final=6"

    # Hacer la solicitud HTTP
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extraer la tabla
    table = soup.find("table", {"class": "table-bordered"})
    rows = table.find_all("tr")
    data = []

    for row in rows:
        cols = row.find_all("td")
        cols = [col.text.strip() for col in cols]
        if cols:
            data.append(cols)

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Eliminar filas vacías y reasignar encabezados
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = ["CONCEPTO"] + [f"Columna_{i}" for i in range(1, df.shape[1])]
    df["CONCEPTO"] = df["CONCEPTO"].str.strip()

    # Filtrar filas relevantes
    diabeticos = df[df["CONCEPTO"] == "DIABETICOS"].iloc[0]
    tabaquismo = df[df["CONCEPTO"] == "TABAQUISMO ≥ 55 AÑOS"].iloc[0]
    enfermedad_renal = df[df["CONCEPTO"].str.contains("ETAPA", na=False)]

    # Extraer columnas numéricas
    def extract_numeric(series):
        return series[1:15].apply(pd.to_numeric, errors='coerce').fillna(0)

    diabeticos_data = extract_numeric(diabeticos)
    tabaquismo_data = extract_numeric(tabaquismo)
    enfermedad_renal_data = enfermedad_renal.iloc[:, 1:15].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=0)

    # Grupos de edad
    grupos_edad = [
        "15 a 19 años", "20 a 24 años", "25 a 29 años", "30 a 34 años",
        "35 a 39 años", "40 a 44 años", "45 a 49 años", "50 a 54 años",
        "55 a 59 años", "60 a 64 años", "65 a 69 años", "70 a 74 años",
        "75 a 79 años", "80 y más años"
    ]

    # Directorio de salida
    output_dir = os.path.join(settings.BASE_DIR, 'static', 'reportes')
    os.makedirs(output_dir, exist_ok=True)

    # Guardar los gráficos
    graficos = []

    # Gráfico 1
    plt.figure(figsize=(12, 6))
    plt.bar(grupos_edad, diabeticos_data, color='blue')
    plt.title('Distribución de Pacientes Diabéticos por Grupo de Edad', fontsize=14)
    plt.xlabel('Grupos de Edad', fontsize=12)
    plt.ylabel('Cantidad de Pacientes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    grafico_1 = os.path.join(output_dir, 'grafico_edad.png')
    plt.savefig(grafico_1)
    plt.close()
    graficos.append(grafico_1)

    # Gráfico 2
    plt.figure(figsize=(12, 6))
    plt.bar(grupos_edad, tabaquismo_data, color='gray', label='Tabaquismo')
    plt.bar(grupos_edad, diabeticos_data, bottom=tabaquismo_data, color='blue', label='Diabéticos')
    plt.title('Comparación entre Tabaquismo y Diabetes por Grupo de Edad', fontsize=14)
    plt.xlabel('Grupos de Edad', fontsize=12)
    plt.ylabel('Cantidad de Personas', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    grafico_2 = os.path.join(output_dir, 'grafico_comparacion.png')
    plt.savefig(grafico_2)
    plt.close()
    graficos.append(grafico_2)

    # Gráfico 3
    plt.figure(figsize=(12, 6))
    plt.bar(grupos_edad, tabaquismo_data, color='gray', label='Tabaquismo')
    plt.bar(grupos_edad, diabeticos_data, bottom=tabaquismo_data, color='blue', label='Diabéticos')
    plt.bar(grupos_edad, enfermedad_renal_data, bottom=(tabaquismo_data + diabeticos_data), color='orange', label='Enfermedad Renal Crónica')
    plt.title('Relación entre Tabaquismo, Diabetes y Enfermedad Renal Crónica', fontsize=14)
    plt.xlabel('Grupos de Edad', fontsize=12)
    plt.ylabel('Cantidad de Personas', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    grafico_3 = os.path.join(output_dir, 'grafico_relacion.png')
    plt.savefig(grafico_3)
    plt.close()
    graficos.append(grafico_3)

    # Gráfico 4
    plt.figure(figsize=(12, 6))
    plt.bar(grupos_edad, enfermedad_renal_data, color='orange')
    plt.title('Progresión de Enfermedad Renal Crónica en Pacientes Diabéticos', fontsize=14)
    plt.xlabel('Grupos de Edad', fontsize=12)
    plt.ylabel('Cantidad de Pacientes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    grafico_4 = os.path.join(output_dir, 'grafico_enfermedad_renal.png')
    plt.savefig(grafico_4)
    plt.close()
    graficos.append(grafico_4)

    return graficos
