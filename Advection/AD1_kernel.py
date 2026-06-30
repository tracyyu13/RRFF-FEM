import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from scipy import io
from RF import *
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

# Write a function to get the training and test samples

def get_data(filename):
    '''
    Read data samples from file.
    
    Input:
    filename.
    
    Returns:
    u0: initial condition (training inputs)
    ut: solution at time t=0.5 (training outputs)
    xt: location x and time t
    '''
    
    # x grid
    nx = 40
    # time grid
    nt = 40
    # load data
    data = np.load(filename)
    # location x, time t, and solution u
    x = data["x"].astype(np.float64)
    t = data["t"].astype(np.float64)
    u = data["u"].astype(np.float64)  # N x nt x nx
    
    # initial condition u0
    u0 = u[:, 0, :]  # N x nx
    # location x and time t
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    
    return u0, u[:, int(nt/2), :], xt

# uses scipy.linalg.solve
class Cholesky_Regression:
    def fit(self, X, y):
        self.coef_ = scipy.linalg.solve(X,y)
        return self

    def predict(self, X):
        return X @ self.coef_
    
# read training data and test data
x_train, y_train, xt= get_data("train_AD1.npz")
x_test, y_test, xt = get_data("test_AD1.npz")

print('Training Input Shape: ' + str(x_train.shape))
print('Training Output Shape: ' + str(y_train.shape))
print('Testing Input Shape: ' + str(x_test.shape))
print('Testing Output Shape: ' + str(y_test.shape) + '\n')

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# add relative noise to x
noise = np.random.normal(0,1,x.shape)
ratio = np.linalg.norm(x, axis=-1)/np.linalg.norm(noise, axis=-1)
ratio = ratio[:,np.newaxis]
x_noise = x + 0.05*noise*ratio

x_train_noise = x_noise[0:x_train.shape[0]]
x_test_noise = x_noise[x_train.shape[0]:]

# add relative noise to y
noise = np.random.normal(0,1,y.shape)
ratio = np.linalg.norm(y, axis=-1)/np.linalg.norm(noise, axis=-1)
ratio = ratio[:,np.newaxis]
y_noise = y + 0.05*noise*ratio

y_train_noise = y_noise[0:y_train.shape[0]]

print('Added noise to data \n')

##################### RBF kernel
bandwidth = 1 # 5
kernel = RBF(length_scale = bandwidth)
model = GaussianProcessRegressor(kernel, alpha = 1e-10)

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
print(f'Gaussian kernel Prediction error is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')

##################### Matern kernel
kernel = Matern(nu = 2.5, length_scale = 2) # nu = 1, length_scale = 50
model = GaussianProcessRegressor(kernel, alpha = 1e-10) 

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()

e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 5 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')
'''
##################### Matern kernel
kernel = Matern(nu = 1.5, length_scale = 50, length_scale_bounds=(1e-20,1e5)) # nu = 1.5, length_scale = 50
model = GaussianProcessRegressor(kernel, alpha = 1e-10) 

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()

e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 3 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')
'''