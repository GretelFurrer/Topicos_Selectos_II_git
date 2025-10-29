# Importación de librerías
import numpy as np
import pprint as pp
import mlflow
import pandas as pd


# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# Módulos propios
from module_data import Dataset # class Dataset
from module_ml import Model



def main():

    data = Dataset()
    X, y = data.load_xy(method='minmax')
    
    # Model
    ml = Model(X=X, y=y, seed=42)
    ml.evaluate(LogisticRegression(max_iter=5000))
    #ml.evaluate(KNeighborsClassifier( n_neighbors=5, weights='uniform', algorithm='auto'))
    ml.evaluate(DecisionTreeClassifier( random_state=10, min_samples_split=50,min_samples_leaf=10))
    ml.evaluate(RandomForestClassifier( n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1  ))
    ml.evaluate(MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=5000, random_state=42 ))
    

if __name__ == "__main__":
    main()
