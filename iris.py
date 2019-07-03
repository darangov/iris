# KNN

# Librerias
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
IRIS_PATH = os.path.join("Diplomado_ML","Diplomado_ML")
# Load the data
def load_iris_data(iris_path=IRIS_PATH):
    csv_path = os.path.join(iris_path, "iris.csv")
    return pd.read_csv(csv_path, header=None) # ya que el dataset no posee encabezado de columnas
'''


# Obtengo los encabezados de una fuente externa
str_encabezados = 'sepal length in cm,sepal width in cm,petal length in cm,petal width in cm,class'
encabezados = str_encabezados.split(',') # separo los encabezados por comas
print("Cantidad encabezados:", len(encabezados)) # valido que la longitud de encabezados corresponda con el numero de columnas de datos

# Cargo el dataset
iris_dataset = pd.read_csv("iris.csv", header=None) # Indico que el dataset original no tiene encabezados o nombre de columnas
# Llamo y asocio las columnas de acuerdo a la lista encabezados
iris_dataset.columns = encabezados # Llamo y asocio las columnas de acuerdo a la lista encabezados

# Top five rows in the dataframe
print()
print ("Top five rows:\n", iris_dataset.head()) 

# Categories related to column 4 - its maybe a Categorical attribute
print("Categorias:\n", iris_dataset.loc[:,'class'].value_counts())

# Description of the data
print()
print ("Data information:\n", iris_dataset.info())

print()
print ("Data describe:\n", iris_dataset.describe())

# Creo mi dataset con X e Y de entrada
X = iris_dataset[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
y = iris_dataset['class'].values

# Creo los sets de entrenamiento y de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print()
print("X_train lenght: ", len(X_train))
print("X_test lenght: ", len(X_test))
print("y_train lenght: ", len(y_train))
print("y_test lenght: ", len(y_test))
print()

# Implemento KNN
n_neighbors = 3
knn_clasificador = KNeighborsClassifier(n_neighbors)
knn_clasificador.fit(X_train, y_train)
print('Accuraccy of K-NN Classifier on training set: {:.2f}'.format(knn_clasificador.score(X_train, y_train)))
print('Accuraccy of K-NN Classifier on test set: {:.2f}'.format(knn_clasificador.score(X_test, y_test)))

"""
# Como la data siempre va a ser la misma (no cambia ni se incrementa o disminuye) usamos la funcion
# train_test_split para usar seed o random state garantizando que siempre genere los mismos indices
#from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris_dataset, test_size = 0.3, random_state = 42)
print()
print(len(train_set), "Train Set data +", len(test_set), "Test Set data")

# Implemento KNN
n_neighbors = 5
knn_clasificador = KNeighborsClassifier(n_neighbors)
knn_clasificador.fit(train_set, test_set)
print('Accuraccy of K-NN Classifier on training set: {:.2f}'.format(knn_clasificador.score(train_set)))
print('Accuraccy of K-NN Classifier on test set: {:.2f}'.format(knn_clasificador.score(test_set)))
"""
