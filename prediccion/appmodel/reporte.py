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

# Ejemplo de uso
print(generar_reporte_vista())
