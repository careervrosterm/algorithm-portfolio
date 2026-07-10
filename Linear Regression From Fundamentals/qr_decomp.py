import numpy as np

def householder(X):
    m, n = X.shape
    Q = np.eye(m) #Creates a square identity matrix of size m
    R = X.copy() #Preserves the original input

    for j in range(0,n-1):
        alpha = np.linalg.norm(R[j:,[j]],2) #The norm of a lower triangular vector is given
        pivot = np.zeros_like(R[j:, [j]]) # A vector with alpha at the j-th index is made (see below)
        pivot[0] = alpha
        u = R[j:, [j]] - pivot #Start of the Householder algorithm. Define u...
        v = u / np.linalg.norm(u,2) #Then v, a normalization of u
        Q_j = np.eye(m-j) - (2 * np.dot(v, v.T)) #This matrix zeroes out all but one element in the j-th column of A
        Q_j = np.block([[np.eye(j), np.zeros((j,m-j))],
                       [np.zeros((m-j,j)), Q_j]]) 
        '''Q_j is changed in size to fit the dimensions of A, with 1 along the upper-left diagonals and 0 elsewhere.'''
        R = np.dot(Q_j, R)
        Q = np.dot(Q, Q_j.T) #Householder matrix is updated (Upper Triangular)

    return Q,R

def qr_solve(X,Y):
    Y_ = np.asarray(Y.copy()).reshape(-1)
    Q, R = householder(X)
    rhs = np.dot(Q.T, Y_)
    beta = np.zeros(R.shape[1])
    for i in reversed(range(R.shape[1])):
        beta[i] = (rhs[i] - np.dot(R[i, i+1:], beta[i+1:])) / R[i, i]
    return beta
        
'''
def decompose(X):
    m,n = X.shape
    for j in range(n): #Select column
        s = np.linalg.norm(X[:,j]) #Euclidean norm of column
        d_j = -s if X[j,j] > 0 else d_j = -s # Define const value based on pivot point sign
        fak = np.sqrt(s * (s + np.abs(X[j,j])))
        X[j,j] = X[j,j] - d_j
        for k in range(j,m):
            X[k,j] = X[k,j] / fak
        for i in range(j+1,n):
            s = 0
            for k in range(j,m): s = s + X[k,j] * X[k,i]
            for k in range(j,m): X[k,i] = X[k,i] - X[k,j]*s
    return X
'''
