import numpy as np

'''This could probably all be conglomerated into one big "error" class whose instance changes the variables, but considering there are only three, this is sufficient.'''
class MSE():
    def D_c(self,n,X,Y,theta_i):
        return -(2/n) * np.dot(X.T,(Y - theta_i))
    def D_i(self,n,Y,theta_i):
        return -(2/n) * np.sum(Y - theta_i)
    def error(self,y, y_hat):
        return np.mean((y - y_hat)**2)

class RMSE():
    def D_c(self,n,X,Y,theta_i):
        return -np.dot(X.T,(Y-theta_i))/(n*RMSE.error(self, Y, theta_i))
    def D_i(self,n,Y,theta_i):
        return np.sum(Y-theta_i)/(n*RMSE.error(self, Y, theta_i))
    def error(self, y, y_hat):
        return np.sqrt(np.mean((y - y_hat)**2))
    
class MAE():
    def D_c(self,n,X,Y,theta_i):
        return -(1/n) * np.dot(X.T, np.sign(Y - theta_i))
    def D_i(self,n,Y,theta_i):
        return -(1/n) * np.sum(np.sign(Y - theta_i))
    def error(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))

'''
def R_squared(y, y_hat):
    y_bar = np.mean(y)
    SStot = np.sum(y - y_bar)
    SSR = np.sum((y - y_hat)**2)
    return 1 - (SSR/SStot)
'''