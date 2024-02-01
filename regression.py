from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from math import *
from scipy.stats import t



class Additif:
    def __init__(self, data_preparation_object):
        self.data_preparation_object = data_preparation_object
        self.model = LinearRegression()

        self.model.fit(data_preparation_object.x_train, data_preparation_object.y_train)

        y_train_predicted = self.model.predict(data_preparation_object.x_train)
        mean_train_absolute_error = np.mean(np.abs (y_train_predicted - data_preparation_object.y_train))
        print(f"Train : {mean_train_absolute_error =:.2f}")

        y_test_predicted = self.model.predict(data_preparation_object.x_test)
        mean_test_absolute_error = np.mean(np.abs(y_test_predicted - data_preparation_object.y_test))
        print(f"Test : {mean_test_absolute_error =:.2f}")

        self.show_model_predictions(y_train_predicted, y_test_predicted)


    

    def show_model_predictions(self, y_train_predicted, y_test_predicted):
        intervalle_conf = 0.95

        residuals = y_test_predicted - self.data_preparation_object.y_test
        std_residuals = np.std(residuals)
        n = len(y_test_predicted)
        margin_of_error = 1.96 * (std_residuals / np.sqrt(n))

        min_interval = (y_test_predicted - margin_of_error).ravel()
        max_interval = (y_test_predicted + margin_of_error).ravel()

        plt.figure(figsize=(15, 6))
        plt.plot(
            self.data_preparation_object.dataset_df['Years'][:len(self.data_preparation_object.x_train)],
            self.data_preparation_object.y_train, "bo-", label='Actual Train Sales')
        plt.plot(
            self.data_preparation_object.dataset_df['Years'][:len(self.data_preparation_object.x_train)],
            y_train_predicted, "b-", label='Predicted Train Sales')

        plt.plot(
            self.data_preparation_object.dataset_df['Years'][len(self.data_preparation_object.x_train):],
            self.data_preparation_object.y_test, "ro-", label='Actual Test Sales')
        plt.plot(
            self.data_preparation_object.dataset_df['Years'][len(self.data_preparation_object.x_train):],
            y_test_predicted, "r-", label='Predicted Test Sales')

        plt.fill_between(self.data_preparation_object.dataset_df['Years'][len(self.data_preparation_object.x_train):],
                         min_interval, max_interval,
                         color="lightgray", alpha=0.5, label="95% Confidence Interval")



        plt.legend()
        plt.show()
