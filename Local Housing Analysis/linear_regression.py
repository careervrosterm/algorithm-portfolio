import numpy as np
from metrics import *

def fit(X,Y,stepsize,metric: str): 
        i = 0
        samples, features = X.shape
        # Begin with the guess of 0 for the coefficients and 0 for the intercept.
        coeff = np.zeros(features)
        intercept = 0
        # Calculate first y_bar output
        theta_i = np.dot(X,coeff) + intercept
        error = error_metric(metric) #One of four metrics from the metrics.py file
        while error.error(Y,theta_i) > 1e-8 and i <= 50000:
            D_c = error.D_c(samples,X,Y,theta_i)  # Derivative wrt coeff
            D_i = error.D_i(samples,Y,theta_i)  # Derivative wrt i
            coeff = coeff - stepsize * D_c  # Update coeff
            intercept = intercept - stepsize * D_i  # Update intercept
            theta_i = np.dot(X,coeff) + intercept #Solve for y using the current iteration guess.
            i += 1
        return coeff, intercept

def error_metric(metric):
    class_list = {
        "MSE": MSE(),
        "MAE": MAE(),
        "RMSE": RMSE()
    }
    return class_list[metric.upper()] 

def predict(X, coeff, intercept):
    return ((X.T*coeff) + intercept)
     

'''
def backtrack(metric): #Backtracking along current guessing path to adapt step size
    beta = 0.5
    return 
'''