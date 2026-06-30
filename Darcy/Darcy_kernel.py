import numpy as np
import scipy
from scipy import io
import time
from RF import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

def get_data(filename, ndata):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29
    r = 15
    s = int(((421 - 1) / r) + 1)

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float64) #* 0.1 - 0.75
    y = data["sol"][:ndata, ::r, ::r].astype(np.float64) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, s * s)
    #x = (x_branch, grid)
    y = y.reshape(ndata, s * s)
    return x_branch, y, grid

# uses scipy.linalg.solve
class Cholesky_Regression:
    def fit(self, X, y):
        self.coef_ = scipy.linalg.solve(X,y)
        return self

    def predict(self, X):
        return X @ self.coef_
    
x_train, y_train, grid = get_data("piececonst_r421_N1024_smooth1.mat", 1000)
print('Loaded training data')

x_test, y_test, grid = get_data("piececonst_r421_N1024_smooth2.mat", 200)
print('Loaded testing data \n')

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
bandwidth = np.sqrt(50000)
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
kernel = Matern(nu = 1, length_scale = 50)
model = GaussianProcessRegressor(kernel, alpha = 1e-10) 

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()

e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 2 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')

##################### Matern kernel
kernel = Matern(nu = 1.5, length_scale = 50)
model = GaussianProcessRegressor(kernel, alpha = 1e-10) 

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()

e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 3 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')