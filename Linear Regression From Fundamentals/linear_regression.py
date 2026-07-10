import numpy as np
from metrics import *

def fit(X,Y,stepsize,metric: str): 
        if X.size == 0 or Y.size == 0:
            raise ValueError("One of the provided matrices is empty.")
        if (X.shape[0] != Y.size and X.shape[1] == 1) or (X.shape != Y.shape):
            raise ValueError("The number of X rows does not match the number of Y elements.")
        A = X.copy()
        b = Y.copy()
        i = 0
        samples, features = A.shape
        # Begin with the guess of 0 for the coefficients and 0 for the intercept.
        coeff = np.zeros([features,1])
        intercept = 0
        # Calculate first y_bar output
        theta_i = np.dot(A,coeff) + intercept
        error = error_metric(metric) #One of four metrics from the metrics.py file
        while error.error(b,theta_i) > 1e-8 and i <= 500000:
            D_c = error.D_c(samples,A,b,theta_i)  # Derivative wrt coeff
            D_i = error.D_i(samples,b,theta_i)  # Derivative wrt i
            coeff = coeff - stepsize * D_c  # Update coeff
            intercept = intercept - stepsize * D_i  # Update intercept
            theta_i = np.dot(A,coeff) + intercept #Solve for y using the current iteration guess.
            i += 1
        return coeff, intercept, error.error(b,theta_i)

def error_metric(metric):
    class_list = {
        "MSE": MSE(),
        "MAE": MAE(),
        "RMSE": RMSE()
    }
    if metric.upper() not in class_list:
         raise ValueError("The metric input does not match any metric classes available.")
    return class_list[metric.upper()] 

def predict(X, coeff, intercept):
    return (np.dot(X,coeff) + intercept)
     

'''
def backtrack(metric): #Backtracking along current guessing path to adapt step size
    beta = 0.5
    return 
'''