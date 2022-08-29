# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 24/08/2022
# Fecha de finalizacion: ?
# ==============================================================================================
#
# ==============================================================================================
# En caso de no tener las bibliotecas necesarias, utilizar los siguientes comandos:
# python -m pip install numpy
# python -m pip install pandas
# python -m pip install scikit-learn
# ==============================================================================================
#
# ==============================================================================================
# Las apoplejias son un evento cuando el suministro de sangre al cerebro se ve interrumpida,
# causando en falta de oxigeno, daño cerebral y perdida de funciones tanto motoras como
# mentales.
# Globalmente, 1 de cada 4 adultos mayores de 25 años va a tener una apoplejia en su vida.
# 12,2 millones de personas tendra su primer apoplejia en este año, y 6.5 millones mas
# moriran como resultado de esta. Mas de 110 millones de personas han tenido una apoplejia. (1)
#
# Este codigo tiene como objetivo analizar datos para poder predecir que personas son mas
# propensas a tener una apoplejia y asi poder evitar secuelas y bajar estas estadisticas.
#
# (1) https://www.world-stroke.org/world-stroke-day-campaign/why-stroke-matters/learn-about-stroke#:~:text=Globally%201%20in%204%20adults,the%20world%20have%20experienced%20stroke.
# ==============================================================================================

# Importamos bibliotecas

import math as m

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# La biblioteca de scikit learn solamente se va a utilizar para modularizar los datos

# Ahora leemos nuestro data-set
# Fuente: https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset

stroke_data = pd.read_csv("Data/full_data.csv")

# Para este analisis y creacion de modelo de regresion logistica, se van a tomar en cuenta
# distintas caracteristicas que son importantes, como la edad, si eran fumadores, enfermedades
# del corazon, etc para crear los modelos.
# Asimismo se van a utilizar caracteristicas que al momento no se consideran relevantes, como
# residencia, genero, estado civil, etc para comparar y ver que tan fuerte es la conexion
# entre caracteristicas que generalmente se consideran relevantes contra caracteristicas
# que normalmente no se considerar relevantes.

# Primera caracteristica: Edad

stroke_x = stroke_data["age"]
stroke_y = stroke_data["stroke"]

# Modularizamos nuestros datos con la ayuda de scikit

stroke_train_x, stroke_test_x, stroke_train_y, stroke_test_y = train_test_split(stroke_x, stroke_y, random_state=1)

# Definimos funcion de hipotesis y funcion de costo

h1  = lambda x, theta: 1 / (1 + m.exp(theta[0] + theta[1] * x))
j_1 = lambda x, y, theta: (y * m.log(h1(x, theta))) + ((1 - y) * (m.log(1 - h1(x, theta))))

# Creacion de modelo de regresion logistica

n = len(stroke_train_y)

alpha = 1e-5

theta = [1, 1]

iteration = 1000

for i in range(iteration):
    accumDelta = []
    accumDeltaX = []

    for x_i, y_i in zip(stroke_train_x, stroke_train_y):
        accumDelta.append(h1(x_i, theta) - y_i)
        accumDeltaX.append((h1(x_i, theta) - y_i) * x_i)

    sJt0 = sum(accumDelta)
    sJt1 = sum(accumDeltaX)

    theta[0] = theta[0] - (alpha/n) * sJt0
    theta[1] = theta[1] - (alpha / n) * sJt1

print("=============================================")
print("Valores de theta para modelo usando 'edad'")
print(theta)
print("=============================================")

# Ahora que ya tenemos nuestro modelo de regresion logistica, ahora lo vamos
# a evaluar con otros datos que no fueron utilizados para entrenar el modelo
# Asimismo, sacaremos los valores necesarios para crear nuestra matriz de
# confusion para este modelo

mat_conf = [[0, 0], [0, 0]]

nj = len(stroke_test_x)
j1 = []

for x_v, y_v in zip(stroke_test_x, stroke_test_y):
    j = j_1(x_v, y_v, theta)
    j1.append(j)

print("Valor de costo promedio")
print(-1 * (sum(j1) / nj))
print("=============================================")

# Ahora sacaremos diferentes metricas para poder medir el rendimiento del modelo
# Estas metricas las usaremos para comparar los diferentes modelos que creemos.

matriz_conf = [[]]


