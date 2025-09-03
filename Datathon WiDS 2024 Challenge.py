#Datathon WiDS 2024 Challenge

#%% Importación de librerías
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score

# %%
df_datathon = pd.read_csv('widsdatathon2024-challenge1/training.csv', sep=',') 
df_datathon.head()
# %%
df_datathon.info()
# %%
