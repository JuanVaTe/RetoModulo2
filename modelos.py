# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 30/08/2022
# Fecha de finalizacion: 31/08/2022
# ==============================================================================================
# Este codigo es un modulo de funciones auxiliares para el archivo 'RetoModulo2.py'

# Bibliotecas
import math as m
import pandas as pd

# Funciones lambda

# Regresion logistica unidimensional
h1 = lambda x, theta: 1 / (1 + m.exp(theta[0] + theta[1] * x))
j_1 = lambda x, y, theta: (y * m.log(h1(x, theta))) + ((1 - y) * (m.log(1 - h1(x, theta))))

# Regresion logistica bidimensional
h2 = lambda x, x2, theta: 1 / (1 + m.exp(theta[0] + theta[1] * x + theta[2] * x2))
j_2 = lambda x, x2, y, theta: (y * m.log(h2(x, x2, theta))) + ((1 - y) * (m.log(1 - h2(x, x2, theta))))

# Regresion logistica tridimensional
h3 = lambda x, x2, x3, theta: 1 / (1 + m.exp(theta[0] + theta[1] * x + theta[2] * x2 + theta[3] * x3))
j_3 = lambda x, x2, x3, y, theta: (y * m.log(h3(x, x2, x3, theta))) + ((1 - y) * (m.log(1 - h3(x, x2, x3, theta))))


# Funcion de regresion logistica unidimensional
def regresion_logistica_unidim(datos_x, datos_y, alpha, iteration, theta):
    n = len(datos_y)

    for i in range(iteration):
        accumDelta = []
        accumDeltaX = []

        for x_i, y_i in zip(datos_x, datos_y):
            accumDelta.append(h1(x_i, theta) - y_i)
            accumDeltaX.append((h1(x_i, theta) - y_i) * x_i)

        sJt0 = sum(accumDelta)
        sJt1 = sum(accumDeltaX)

        theta[0] = theta[0] - (alpha / n) * sJt0
        theta[1] = theta[1] - (alpha / n) * sJt1

    return theta


# Funcion de validacion de regresion logistica unidimensional con matriz de confusion
def funcion_costo_unidimensional(datos_x, datos_y, theta):
    matriz_confusion = [[0, 0], [0, 0]]

    nj = len(datos_y)
    j1 = []

    for x_v, y_v in zip(datos_x, datos_y):
        j = j_1(x_v, y_v, theta)
        j1.append(j)

        prediction = round(h1(x_v, theta))

        if (prediction == y_v) and (prediction == 1):
            matriz_confusion[0][0] = matriz_confusion[0][0] + 1

        if (prediction != y_v) and (prediction == 0):
            matriz_confusion[0][1] = matriz_confusion[0][1] + 1

        if (prediction != y_v) and (prediction == 1):
            matriz_confusion[1][0] = matriz_confusion[1][0] + 1

        if (prediction == y_v) and (prediction == 0):
            matriz_confusion[1][1] = matriz_confusion[1][1] + 1

    costo_promedio = -1 * (sum(j1) / nj)

    return costo_promedio, matriz_confusion


# Funcion de regresion logistica bidimensional
def regresion_logistica_bidim(datos_x, datos_y, alpha, iteration, theta):
    n = len(datos_y)

    for i in range(iteration):
        accumDelta = []
        accumDeltaX = []
        accumDeltaX2 = []

        for x_i, x2_i, y_i in zip(datos_x.iloc[:, 0], datos_x.iloc[:, 1], datos_y):
            accumDelta.append(h2(x_i, x2_i, theta) - y_i)
            accumDeltaX.append((h2(x_i, x2_i, theta) - y_i) * x_i)
            accumDeltaX2.append((h2(x_i, x2_i, theta) - y_i) * x2_i)

        sJt0 = sum(accumDelta)
        sJt1 = sum(accumDeltaX)
        sJt2 = sum(accumDeltaX2)

        theta[0] = theta[0] - (alpha / n) * sJt0
        theta[1] = theta[1] - (alpha / n) * sJt1
        theta[2] = theta[2] - (alpha / n) * sJt2

    return theta


# Funcion de validacion de regresion logistica bidimensional con matriz de confusion
def funcion_costo_bidimensional(datos_x, datos_y, theta):
    matriz_confusion = [[0, 0], [0, 0]]

    nj = len(datos_y)
    j2 = []

    for x_v, x2_v, y_v in zip(datos_x.iloc[:, 0], datos_x.iloc[:, 1], datos_y):
        j = j_2(x_v, x2_v, y_v, theta)
        j2.append(j)

        prediction = round(h2(x_v, x2_v, theta))

        if (prediction == y_v) and (prediction == 1):
            matriz_confusion[0][0] = matriz_confusion[0][0] + 1

        if (prediction != y_v) and (prediction == 0):
            matriz_confusion[0][1] = matriz_confusion[0][1] + 1

        if (prediction != y_v) and (prediction == 1):
            matriz_confusion[1][0] = matriz_confusion[1][0] + 1

        if (prediction == y_v) and (prediction == 0):
            matriz_confusion[1][1] = matriz_confusion[1][1] + 1

    costo_promedio = -1 * (sum(j2) / nj)

    return costo_promedio, matriz_confusion


# Funcion de regresion logistica tridimensional
def regresion_logistica_tridim(datos_x, datos_y, alpha, iteration, theta):
    n = len(datos_y)

    for i in range(iteration):
        accumDelta = []
        accumDeltaX = []
        accumDeltaX2 = []
        accumDeltaX3 = []

        for x_i, x2_i, x3_i, y_i in zip(datos_x.iloc[:, 0], datos_x.iloc[:, 1], datos_x.iloc[:, 2], datos_y):
            accumDelta.append(h3(x_i, x2_i, x3_i, theta) - y_i)
            accumDeltaX.append((h3(x_i, x2_i, x3_i, theta) - y_i) * x_i)
            accumDeltaX2.append((h3(x_i, x2_i, x3_i, theta) - y_i) * x2_i)
            accumDeltaX3.append((h3(x_i, x2_i, x3_i, theta) - y_i) * x3_i)

        sJt0 = sum(accumDelta)
        sJt1 = sum(accumDeltaX)
        sJt2 = sum(accumDeltaX2)
        sJt3 = sum(accumDeltaX3)

        theta[0] = theta[0] - (alpha / n) * sJt0
        theta[1] = theta[1] - (alpha / n) * sJt1
        theta[2] = theta[2] - (alpha / n) * sJt2
        theta[3] = theta[3] - (alpha / n) * sJt3

    return theta


# Funcion de validacion de regresion logistica tridimensional con matriz de confusion
def funcion_costo_tridimensional(datos_x, datos_y, theta):
    matriz_confusion = [[0, 0], [0, 0]]

    nj = len(datos_y)
    j3 = []

    for x_v, x2_v, x3_v, y_v in zip(datos_x.iloc[:, 0], datos_x.iloc[:, 1], datos_x.iloc[:, 2], datos_y):
        j = j_3(x_v, x2_v, x3_v, y_v, theta)
        j3.append(j)

        prediction = round(h3(x_v, x2_v, x3_v, theta))

        if (prediction == y_v) and (prediction == 1):
            matriz_confusion[0][0] = matriz_confusion[0][0] + 1

        if (prediction != y_v) and (prediction == 0):
            matriz_confusion[0][1] = matriz_confusion[0][1] + 1

        if (prediction != y_v) and (prediction == 1):
            matriz_confusion[1][0] = matriz_confusion[1][0] + 1

        if (prediction == y_v) and (prediction == 0):
            matriz_confusion[1][1] = matriz_confusion[1][1] + 1

    costo_promedio = -1 * (sum(j3) / nj)

    return costo_promedio, matriz_confusion


# Funcion de metricas de rendimiento
def metricas_rendimiento(matriz_confusion):
    exactitud = (matriz_confusion[0][0] + matriz_confusion[1][1]) / (
                matriz_confusion[0][0] + matriz_confusion[0][1] + matriz_confusion[1][0] + matriz_confusion[1][1])

    try:
        precision = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[1][0])
    except:
        precision = 0

    exhaustividad = matriz_confusion[0][0] / (matriz_confusion[0][0] + matriz_confusion[0][1])

    try:
        puntaje_F1 = (2 * precision * exhaustividad) / (precision + exhaustividad)
    except:
        puntaje_F1 = 0

    return exactitud, precision, exhaustividad, puntaje_F1
