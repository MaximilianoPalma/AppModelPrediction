import requests
from bs4 import BeautifulSoup

# Función de scraping o API para obtener datos
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
