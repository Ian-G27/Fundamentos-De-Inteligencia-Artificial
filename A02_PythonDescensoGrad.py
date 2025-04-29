#=============================================
# Descenso de Gradiente para Mínimos Cuadrados
#=============================================
# Ian Diego Buendia Alvarez
# ESFM IPN
# Febrero 2025
#=============================================
#Regresion lineal
#=============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from A01_PythonRegresionLineal import minimos_cuadrados
#========================================
# Descenso de gradiente simple
#========================================
def descensoG(epocs,X,Y,alpha):
   w0 = 0.0
   wl = 0.0
   N = len(X)
   sumx = np.sum(X)
   sumy = np.sum(Y)
   sumxy = np.sum(X * Y)
   sumx2 = np.sum(X * X)
   for i in range(epocs):
     Gradw0 = -2.0 * (sumy - w0 * N - wl * sumx)
     Gradwl = -2.0 * (sumxy - w0 * sumx - wl * sumx2)
     w0 -= alpha*Gradw0
     wl -= alpha*Gradwl
   return w0, wl
#==============================================
#   Programa principal
#==============================================
if "__main__" == __name__:
  #========================
  # Leer datos
  #========================
  data = pd.read_csv('data.csv')
  X = np.array(data.iloc[:, 0])
  Y = np.array(data.iloc[:, 1])
  #======================
  # Mínimos cuadrados
  #======================
  Ybar,w0,wl = minimos_cuadrados(X, Y)
  #===========================
  # Descenso de gradiente
  #===========================
  w0 = 0.0
  wl = 0.0
  alpha = 0.025
  epocs = 150
  w0, wl = descensoG(epocs, X, Y, alpha)
  Ybar2 = w0 + wl*X
  #=================
  # Gráfica
  #=================
  plt.scatter(X, Y)
  plt.rcParams['figure.figsize'] = (12.0, 9.0)
  plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')
  plt.plot([min(X), max(X)], [min(Ybar2), max(Ybar2)], color='green')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()