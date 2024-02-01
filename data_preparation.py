from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from math import *




class DataPreparation:
    def __init__(self, dataset_df):
        self.dataset_df = dataset_df
        """
        Cette classe prend en entrée un chemin de fichier csv.
        Elle split le jeu de donnée en 2 bases 
        + une train 75 %
        + une test 25 %
        Ce 2 bases, la classe va les splits en 2 

        + un vecteur x (qui contient les indexs temporels)
        + un vecteur y (qui contient les valeurs à prédire)
        En tout cette va extraire 4 arrays.
        x_train
        y_train
        x_test
        y_test
        """

        self.dataset_df = pd.read_csv('vente_maillots_de_bain.csv', encoding='ISO-8859-1')

        self.dataset_df['Years'] = pd.to_datetime(self.dataset_df['Years'])
        self.dataset_df['month_name'] = self.dataset_df['Years'].dt.strftime('%B')

        self.dataset_df = pd.get_dummies(self.dataset_df, columns=['month_name'], drop_first=True)
        self.dataset_df.replace({True: 1, False: 0}, inplace=True)
        self.dataset_df['index_interation'] = np.arange(1, len(self.dataset_df) + 1)

        self.dataset_df = pd.DataFrame(self.dataset_df)
        self.prepare_data()
        dataset_df
        


    def prepare_data(self):
        number_of_rows = len(self.dataset_df)
        self.dataset_df["month_name"] = np.arange(0, number_of_rows, 1)

        dataset_train_df = self.dataset_df.iloc[:int(number_of_rows * 0.75)]
        dataset_test_df = self.dataset_df.iloc[int(number_of_rows * 0.75):]

        self.x_train = dataset_train_df.iloc[:, 3:14].values
        self.y_train = dataset_train_df[['Sales']].values

        self.x_test = dataset_test_df.iloc[:, 3:14].values
        self.y_test = dataset_test_df[['Sales']].values