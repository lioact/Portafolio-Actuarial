# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:35:41 2024

@author: leo_m
"""

import pandas as pd

# Ruta del archivo Excel
datos = pd.read_excel(r"C:\Users\leo_m\OneDrive - Benemérita Universidad Autónoma de Puebla\Universidad\8° semestre\Estadistica 2\Proyecto\TODAS.xlsx")

# Diccionarios de mapeo
mapeo_likert_general = {
    "Muy insatisfecho(a)": 1,
    "Insatisfecho(a)": 2,
    "Neutral": 3,
    "Satisfecho(a)": 4,
    "Muy satisfecho(a)": 5
}

mapeo_relevancia = {
    "Totalmente en desacuerdo": 1,
    "En desacuerdo": 2,
    "Neutral": 3,
    "De acuerdo": 4,
    "Totalmente de acuerdo": 5
}

mapeo_recomiendas = {
    "Muy improbable": 1,
    "Improbable": 2,
    "Neutral": 3,
    "Probable": 4,
    "Muy probable": 5
}


mapeo_horas = {
    "Menos de 5": 1,
    "5-10 horas": 2,
    "11-15 horas": 3,
    "16-20 horas": 4,
    "Más de 20 horas": 5
}

# Aplicar mapeo a la columna HORAS
if 'HORAS' in datos.columns:
    datos['HORAS'] = datos['HORAS'].map(mapeo_horas)

# Columnas a transformar y sus diccionarios
columnas_transformar = {
    "ENSEÑANZA": mapeo_likert_general,
    "RECURSOS": mapeo_likert_general,
    "APOYO": mapeo_likert_general,
    "RELEVANCIA": mapeo_relevancia,
    "RECOMIENDAS": mapeo_recomiendas
}

# Transformar las columnas
for columna, mapeo in columnas_transformar.items():
    if columna in datos.columns:
        datos[columna] = datos[columna].astype(str).str.strip()  # Limpia espacios y asegura cadenas
        datos[columna] = datos[columna].map(mapeo)

# Guarda el archivo procesado
ruta_salida = r"C:\Users\leo_m\OneDrive - Benemérita Universidad Autónoma de Puebla\Universidad\8° semestre\Estadistica 2\Proyecto\archivo_procesado.xlsx"
datos.to_excel(ruta_salida, index=False)

print(f"Archivo procesado guardado como: {ruta_salida}")


















