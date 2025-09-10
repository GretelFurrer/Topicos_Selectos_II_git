#Importacion de librarias
import numpy as np
from sklearn.linear_model import LogisticRegression

#Modulos propios
from module_data import Dataset #class Dataset
from module_ml import Model

def main():
    data = Dataset()
    df_train, df_test = data.load_data_clean_encoded()
    print(df_train.head())

    #model
    ml = Model(X=X,y=y, seed = 42)
    ml.evaluate(LogisticRegression(max_iter=5000))

if __name__ == '__main__':
    main()