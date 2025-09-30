# Importación de librerías
import numpy as np
import pprint as pp

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
    ml.evaluate(DecisionTreeClassifier( max_depth=5, min_samples_split=2, min_samples_leaf=1 ))
    ml.evaluate(RandomForestClassifier( n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1  ))

if __name__ == "__main__":
    main()
