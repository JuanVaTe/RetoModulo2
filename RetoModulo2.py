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

#Importamos bibliotecas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#La biblioteca de scikit learn solamente se va a utilizar para modularizar los datos

#Ahora leemos nuestro data-set

stroke_data = pd.read_csv("Data/full_data.csv")


