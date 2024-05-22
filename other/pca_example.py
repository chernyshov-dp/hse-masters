import numpy as np
import matplotlib.pyplot as plt


def PCA(data):
    """
    PCA: Perform PCA using covariance.
    data - MxN matrix of input data
    (M dimensions, N trials)
    signals - MxN matrix of projected data
    PC - each column is a PC
    V - Mx1 matrix of variances
    """

    M, N = data.shape
    
    # subtract off the mean for each dimension
    mn = np.mean(data, axis=1)
    data = data - np.tile(mn[:, np.newaxis], (1, N))
    
    # calculate the covariance matrix
    covariance = np.dot(data, data.T) / (N - 1)
    
    # find the eigenvalues and eigenvectors
    PC, V = np.linalg.eig(covariance)
    
    # sort the variances in decreasing order
    rindices = np.argsort(-1 * V)
    V = V[rindices]
    PC = PC[rindices]
    
    # project the original data set
    signals = np.dot(PC.T, data)
    
    return signals, PC, V

np.random.seed(0)
data = np.random.randn(2, 100)

signals, PC, V = PCA(data)

plt.scatter(data[0, :], data[1, :], alpha=0.5)
plt.title("Original Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

plt.scatter(signals[0, :], signals[1, :], alpha=0.5)
plt.title("Data after PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
