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
import modelos

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

# Nuestras funciones de hipotesis y costo se encuentran en el modulo de 'modelos.py'
# Se utilizaron los siguientes:
# h1 = lambda x, theta: 1 / (1 + m.exp(theta[0] + theta[1] * x))
# j_1 = lambda x, y, theta: (y * m.log(h1(x, theta))) + ((1 - y) * (m.log(1 - h1(x, theta))))

# Implementacion del modelo de regresion logistica
# La funcion de regresion logistica se encuentra en el modulo de 'modelos.py'

theta_edad = [1, 1]

alpha = 1e-5

theta_edad = modelos.regresion_logistica_unidim(stroke_train_x, stroke_train_y, alpha=alpha, iteration=1000, theta=theta_edad)

# Impresion de resultados
print("=============================================")
print("Valores de theta para modelo usando 'edad'")
print(theta_edad)

# Ahora que ya tenemos nuestro modelo de regresion logistica lo vamos
# a evaluar con otros datos que no fueron utilizados para entrenar el modelo
# Asimismo, sacaremos los valores necesarios para crear nuestra matriz de
# confusion para este modelo
# (Los positivos seran los 1's y los negativos los 0's)
# La implementacion de la funcion de costo y obtencion de matriz de confusion se
# encuentran en el modulo 'modelos.py'

costo_promedio_edad, mat_conf_edad = modelos.funcion_costo_unidimensional(stroke_test_x, stroke_test_y, theta_edad)

print("=============================================")
print("Valor de costo promedio")
print(costo_promedio_edad)
print("=============================================")
print("Matriz de confusion")
print(mat_conf_edad)

# Ahora sacaremos diferentes metricas para poder medir el rendimiento del modelo
# Estas metricas las usaremos para comparar los diferentes modelos que creemos.
# La obtencion de las metricas de rendimiento se encuentra en el modulo 'modelos.py'

exactitud_edad, precision_edad, exhaustividad_edad, puntaje_F1_edad = modelos.metricas_rendimiento(mat_conf_edad)

print("=============================================")
print("Metricas de rendimiento para caracteristica 'edad'")
print(f"Exactitud     : {exactitud_edad}")
print(f"Precision     : {precision_edad}")
print(f"Exhaustividad : {exhaustividad_edad}")
print(f"Puntaje F1    : {puntaje_F1_edad}")
print("=============================================\n\n\n")


# Aunque la exactitud nos diga que nuestro modelo puede llegar a ser bueno, todas las
# demas metricas se encuentran en el suelo. Nuestro modelo predice que absolutamente
# todas las personas no tienen riesgo de apoplejia.

# Segunda caracteristica: fumadores

stroke_x = stroke_data["smoking_status"]
stroke_y = stroke_data["stroke"]

# Antes de poder empezar a implementar un modelo de regresion logistica, debemos
# convertir nuestros datos cualitativos a datos numericos binarios. Para esto,
# pandas cuenta con una funcion que nos puede ayudar.

# Primero vemos los datos unicos con los que cuenta nuestro data-frame

print("=============================================")
print("Valores unicos en caracteristca de 'fumador'")
print(pd.unique(stroke_x))

# Ya que tenemos datos incompletos, vamos a quitarlos para no contaminar nuestro
# modelo de regresion
# Tambien tendremos que quitar estos datos de nuestra variable dependiente para
# que se mantengan con la misma cantidad de datos

stroke_data_smoking = pd.concat([stroke_x, stroke_y], axis=1)
stroke_data_smoking = stroke_data_smoking[stroke_data_smoking['smoking_status'] != 'Unknown'].reset_index(drop=True)

# Debido a que la variable cualitativa 'smoking_status' tiene un dominio mayor
# de 2, es necesario crear una columna para cada valor unico.
# Afortunadamente pandas cuenta con una funcon para ayudarnos con eso

stroke_data_smoking = pd.concat([pd.get_dummies(stroke_data_smoking['smoking_status'], drop_first=True), stroke_data_smoking], axis=1).drop('smoking_status', axis=1)

# Ahora, vamos a separar nuevamente este nuevo data-frame para obtener nuestras variables
# independientes y dependientes

stroke_x = stroke_data_smoking.drop('stroke', axis=1)
stroke_y = stroke_data_smoking['stroke']

# Por ultimo, vamos a modularizar para que nuestro modelo sea validado con datos que no se
# utilizaron para entrenarlo

stroke_train_x, stroke_test_x, stroke_train_y, stroke_test_y = train_test_split(stroke_x, stroke_y, random_state=1)

# Ahora vamos a implementar un modelo de regresion logistica. Puede que la primera impresion
# nos indique que, ya que fueron 3 valores posibles los que pudo tomar la variable de 'smoking_status',
# la realidad es que pandas nos hizo solamente 2 columnas. Pero no es de preocuparse ya que
# para representar a la tercera posibilidad, ambos valores que tienen una columna deben de tener un
# 0. De esta forma nos ahorramos mucho poder computacional sin sacrificar nada.

# Ahora implementaremos una regresion logistica bidimensional ('modelos.py')

theta_smoking = [1, 1, 1]

alpha = 1e-5

theta_smoking = modelos.regresion_logistica_bidim(stroke_train_x, stroke_train_y, alpha=alpha, iteration=10000, theta=theta_smoking)

# Impresion de resultados
print("=============================================")
print("Valores de theta para modelo usando 'smoking'")
print(theta_smoking)

# Ahora sacaremos el costo promedio de la funcion de hipotesis
# junto con su matriz de confusion ('modelos.py')

costo_promedio_smoking, mat_conf_smoking = modelos.funcion_costo_bidimensional(stroke_test_x, stroke_test_y, theta_smoking)

print("=============================================")
print("Valor de costo promedio")
print(costo_promedio_edad)
print("=============================================")
print("Matriz de confusion")
print(mat_conf_edad)