import numpy as np
from metrics import *

def fit(X,Y,stepsize,metric: str): 
        i = 0
        n = len(X)
        # Begin with the guess of 0 for the coefficients and 0 for the intercept.
        coeff = np.zeros(n)
        intercept = 0
        # Calculate first y_bar output
        theta_i = np.matmul(coeff,X) + intercept
        error = error_metric(metric) #One of four metrics from the metrics.py file
        while error.error(Y,theta_i) < 1e-8 or i <= 50000:
            D_c = error.D_c(n,X,Y,theta_i)  # Derivative wrt coeff
            D_i = error.D_i(n,Y,theta_i)  # Derivative wrt i
            coeff = coeff - stepsize * D_c  # Update coeff
            intercept = intercept - stepsize * D_i  # Update intercept
            theta_i = np.matmul(coeff,X) + intercept #Solve for y using the current iteration guess.
            i += 1
        err = error.error(Y,theta_i)
        return coeff, intercept, err

def error_metric(metric):
    class_list = {
        "MSE": MSE(),
        "MAE": MAE(),
        "RMSE": RMSE()
    }
    return class_list[metric.upper()] 

'''
def backtrack(metric): #Backtracking along current guessing path to adapt step size
    beta = 0.5
    return 
'''