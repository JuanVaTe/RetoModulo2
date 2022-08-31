# ==============================================================================================
# Autor: Juan Carlos Varela Tellez
# Fecha de inicio: 24/08/2022
# Fecha de finalizacion: 31/08/2022
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

import time
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

# Primera caracteristica 'edad'

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

theta_smoking = modelos.regresion_logistica_bidim(stroke_train_x, stroke_train_y, alpha=alpha, iteration=500, theta=theta_smoking)

# Impresion de resultados
print("=============================================")
print("Valores de theta para modelo usando 'smoking'")
print(theta_smoking)

# Ahora sacaremos el costo promedio de la funcion de hipotesis
# junto con su matriz de confusion ('modelos.py')

costo_promedio_smoking, mat_conf_smoking = modelos.funcion_costo_bidimensional(stroke_test_x, stroke_test_y, theta_smoking)

print("=============================================")
print("Valor de costo promedio")
print(costo_promedio_smoking)
print("=============================================")
print("Matriz de confusion")
print(mat_conf_smoking)

# Ahora sacaremos las metricas de rendimiento de este modelo ('modelos.py')

exactitud_smoking, precision_smoking, exhaustividad_smoking, puntaje_F1_smoking = modelos.metricas_rendimiento(mat_conf_smoking)

print("=============================================")
print("Metricas de rendimiento para caracteristica 'smoking'")
print(f"Exactitud     : {exactitud_smoking}")
print(f"Precision     : {precision_smoking}")
print(f"Exhaustividad : {exhaustividad_smoking}")
print(f"Puntaje F1    : {puntaje_F1_smoking}")
print("=============================================\n\n\n")


# Desafortunadamente, este modelo tiene el mismo rendimiento que el anterior.
# Lo que podemos inferir viendo el data-set con el cual estamos trabajando
# es que la mayoria de los datos indican que la persona no tuvo apoplejia, entonces
# nuestros modelos aprenden eso. Esto es ya que nuestros modelos son simples
# y se necesitaria un mejor metodo o modelo para poder empezar a predecir de
# forma correcta.

# Tercera caracteristica: 'residence_type' (no relevante)

# La intuicion nos indica que esta caracteristica no deberia tener tanto peso
# como la edad o si la persona fumaba. Para poder asegurarnos de esto, vamos
# a hacer un modelo de regresion logistica para comparar su rendimiento

stroke_x = stroke_data["Residence_type"]
stroke_y = stroke_data["stroke"]

# Ahora vemos los datos unicos con los que cuenta nuestro data-frame

print("=============================================")
print("Valores unicos en caracteristca de 'Residence_type'")
print(pd.unique(stroke_x))

# Afortunadamente, solamente hay 2 posibilidades que nuestra caracteristica
# puede tomar. entonces vamos a cambiar nuestra columna a una columna binaria
# para poder ser utilizada por nuestro modelo de regresion logistica

stroke_x = pd.get_dummies(stroke_x, drop_first=True)
stroke_x = pd.Series(stroke_x['Urban'])  # Regresarlo a una serie de pandas en vez de un data-frame

# Despues, vamos a modularizar nuestros datos para poder validarlo con
# datos que no fueron utilizados para entrenar al modelo

stroke_train_x, stroke_test_x, stroke_train_y, stroke_test_y = train_test_split(stroke_x, stroke_y, random_state=1)

# Es tiempo de implementar un modelo de regresion logistica unidimensional ('modelos.py')

theta_residence = [1, 1]

alpha = 1e-5

theta_residence = modelos.regresion_logistica_unidim(stroke_train_x, stroke_train_y, alpha=alpha, iteration=500, theta=theta_residence)

# Impresion de resultados
print("=============================================")
print("Valores de theta para modelo usando 'residence'")
print(theta_residence)

# Ahora sacaremos el costo promedio de la funcion de hipotesis
# junto con su matriz de confusion ('modelos.py')

costo_promedio_residence, mat_conf_residence = modelos.funcion_costo_unidimensional(stroke_test_x, stroke_test_y, theta_residence)

print("=============================================")
print("Valor de costo promedio")
print(costo_promedio_residence)
print("=============================================")
print("Matriz de confusion")
print(mat_conf_residence)

# Ahora sacaremos las metricas de rendimiento de este modelo ('modelos.py')

exactitud_residence, precision_residence, exhaustividad_residence, puntaje_F1_residence = modelos.metricas_rendimiento(mat_conf_residence)

print("=============================================")
print("Metricas de rendimiento para caracteristica 'residence'")
print(f"Exactitud     : {exactitud_residence}")
print(f"Precision     : {precision_residence}")
print(f"Exhaustividad : {exhaustividad_residence}")
print(f"Puntaje F1    : {puntaje_F1_residence}")
print("=============================================\n\n\n")

# Podemos observar el mismo fenomeno donde nuestro modelo predice que
# no hay peligro alguno de apoplejia. Aunque estamos utilizando una variable
# poco relevante, esto nos indica que podriamos necesitar un modelo mas complejo.

# Modelo combinado: 'edad' y 'smoking'

# Para poder crear un modelo mas complejo, necesitamos contemplar mas caracteristicas,
# es por eso que vamos a utilizar las caracteristicas que consideramos mas relevantes
# y vamos a usarlas al mismo tiempo en un nuevo modelo

stroke_data_comb = stroke_data[['age', 'smoking_status', 'stroke']]

# Vamos a eliminar nuevamente las filas con un valor de 'Unknown' en la columna
# de 'smoking_status'

stroke_data_comb = stroke_data_comb[stroke_data_comb['smoking_status'] != 'Unknown'].reset_index(drop=True)

# Ahora sacaremos nuevamente las columnas necesarias para poder utilizar la variable
# cualitativa de 'smoking_status'

stroke_data_comb = pd.concat([pd.get_dummies(stroke_data_comb['smoking_status'], drop_first=True), stroke_data_comb], axis=True).drop('smoking_status', axis=True)

# Con este proceso ya podemos obtener los datos para poder empezar a implementar nuestro
# modelo de regresion

stroke_x = stroke_data_comb.drop('stroke', axis=1)
stroke_y = stroke_data_comb['stroke']

# Modularizamos las variables para validar el modelo sin utilizar datos que fueron usados
# para entrenar al modelo

stroke_train_x, stroke_test_x, stroke_train_y, stroke_test_y = train_test_split(stroke_x, stroke_y, random_state=1)

# Implementamos un modelo de regresion logistica tridimensional

theta_comb = [1, 1, 1, 1]

alpha = 1e-5

theta_comb = modelos.regresion_logistica_tridim(stroke_train_x, stroke_train_y, alpha=alpha, iteration=500, theta=theta_comb)

# Impresion de resultados
print("=============================================")
print("Valores de theta para modelo usando 'edad' y 'smoking'")
print(theta_comb)

# Ahora sacaremos el costo promedio de la funcion de hipotesis
# junto con su matriz de confusion ('modelos.py')

costo_promedio_comb, mat_conf_comb = modelos.funcion_costo_tridimensional(stroke_test_x, stroke_test_y, theta_comb)

print("=============================================")
print("Valor de costo promedio")
print(costo_promedio_comb)
print("=============================================")
print("Matriz de confusion")
print(mat_conf_comb)

# Ahora sacaremos las metricas de rendimiento de este modelo ('modelos.py')

exactitud_comb, precision_comb, exhaustividad_comb, puntaje_F1_comb = modelos.metricas_rendimiento(mat_conf_comb)

print("=============================================")
print("Metricas de rendimiento para caracteristica 'residence'")
print(f"Exactitud     : {exactitud_comb}")
print(f"Precision     : {precision_comb}")
print(f"Exhaustividad : {exhaustividad_comb}")
print(f"Puntaje F1    : {puntaje_F1_comb}")
print("=============================================")

# Incluso utilizando diferentes caracteristicas, si el modelo sigue siendo simple,
# vemos que tiene sus limitaciones. Para poder resolver y predecir complicaciones
# como estas es necesario utilizar metodos mas optimos y poderosos.

# Conclusion

# Es importante no casarse con solamente un tipo de modelo para resolver distintos
# problemas. Aunque en este codigo se utilizaron modelos muy basicos, si se intenta
# resolver todos los problemas con la misma herramienta, no importa lo robusta o
# poderosa, no siempre va a dar el mejor resultado.

# EXTRA: comparacion con biblioteca scikit-learn

# Vamos a ver que tanto erro nuestro modelo en comparacion con un modelo optimizado

# Importamos regresion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# La implementamos con nuestros datos que utilizamos en el ultimo modelo

logisticRegModel = LogisticRegression(class_weight='balanced')
logisticRegModel.fit(stroke_train_x, stroke_train_y)

# Sacamos las metricas para poder comparar con los otros modelos

predicciones = logisticRegModel.predict(stroke_test_x)

print("Uso de sklearn.linear_model-LogisticRegression")
print(classification_report(stroke_test_y, predicciones))

# Para poder obtener mejores resultados, en cuanto a las personas que si
# son propensas a apoplejias, es necesario mover las metricas de la
# regresion logistica para ver cual opcion seria mejor para este problema
# en especifico.
