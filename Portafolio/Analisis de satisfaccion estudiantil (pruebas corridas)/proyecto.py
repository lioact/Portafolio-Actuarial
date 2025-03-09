# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:14:35 2024

@author: leo_m
"""

import pandas as pd
from scipy.stats import norm



import seaborn as sns
import matplotlib.pyplot as plt
 
    

# Cargar el archivo procesado
datos = pd.read_excel(r"C:\Users\leo_m\OneDrive - Benemérita Universidad Autónoma de Puebla\Universidad\8° semestre\Estadistica 2\Proyecto\archivo_procesado.xlsx")



# Función para la prueba de corridas
def prueba_corridas(serie):
    # Obtener la mediana de la serie
    mediana = serie.median()
    
    # Convertir la serie en una secuencia binaria (1 si >= mediana, 0 si < mediana)
    binaria = serie.apply(lambda x: 1 if x >= mediana else 0)
    
    # Contar el número de corridas observadas
    corridas = 1 + sum(binaria[i] != binaria[i-1] for i in range(1, len(binaria)))
    
    # Contar el número de unos y ceros
    n1 = sum(binaria)
    n2 = len(binaria) - n1
    
    # Calcular la media y varianza de las corridas bajo la hipótesis nula
    media_corridas = (2 * n1 * n2) / (n1 + n2) + 1
    varianza_corridas = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    
    # Calcular el estadístico Z
    z = (corridas - media_corridas) / (varianza_corridas**0.5)
    
    # Valor p
    p_valor = 2 * (1 - norm.cdf(abs(z)))  # Prueba bilateral
    
    return {
        "Mediana": mediana,
        "Corridas observadas": corridas,
        "Media esperada": media_corridas,
        "Varianza esperada": varianza_corridas,
        "Estadístico Z": z,
        "Valor p": p_valor
    }

# Aplicar la prueba a las columnas indicadas
columnas_a_probar = ['HORAS','ENSEÑANZA', 'RECURSOS', 'APOYO', 'RELEVANCIA', 'RECOMIENDAS']

resultados = {}
for columna in columnas_a_probar:
    if columna in datos.columns:
        resultados[columna] = prueba_corridas(datos[columna])

# Mostrar resultados
for columna, resultado in resultados.items():
    print(f"Resultados para la columna {columna}:")
    for clave, valor in resultado.items():
        print(f"  {clave}: {valor}")
    print("\n")



########################################################################









# Filtro por carreras
carreras_interes = [
    "Actuaría",
    "Física",
    "Física Aplicada",
    "Matemáticas",
    "Matemáticas Aplicadas"
]

# Seleccionar columnas relevantes
columnas_interes = ['SEMESTRE ', ' PROMEDIO', 'HORAS', 'ENSEÑANZA','RECURSOS', 'APOYO','RELEVANCIA','RECOMIENDAS']

# Generar un mapa de calor para cada carrera
for carrera in carreras_interes:
    # Filtrar los datos para la carrera actual
    datos_carrera = datos[datos['CARRERA'] == carrera][columnas_interes].dropna()

    # Verificar si hay suficientes datos
    if datos_carrera.shape[0] > 1:  # Necesitamos al menos dos filas para calcular correlaciones
        # Calcular matriz de correlación de Spearman
        matriz_corr = datos_carrera.corr(method='spearman')

        # Crear mapa de calor
        plt.figure(figsize=(8, 6))
        sns.heatmap(    
            matriz_corr,
            annot=True,        # Mostrar valores en cada celda
            cmap="coolwarm",   # Paleta de colores
            cbar=True,         # Mostrar barra de color
            square=True        # Celdas cuadradas
        )
        plt.title(f"Mapa de Calor - Correlación de Spearman ({carrera})", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No hay suficientes datos para la carrera {carrera}.")
































