import numpy as np
import scipy
import time
from scipy import io
from RF import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

def get_data(ntrain, ntest):
    sub_x = 2 ** 6
    sub_y = 2 ** 6

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat("burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(np.float64)
    y_data = data["u"][:, ::sub_y].astype(np.float64)
    x_branch_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_branch_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]
    
    s = 2 ** 13 // sub_y  # total grid size divided by the subsampling rate
    grid = np.linspace(0, 1, num=2 ** 13)[::sub_y, None]
    
    return x_branch_train, y_train, x_branch_test, y_test, grid

# uses scipy.linalg.solve
class Cholesky_Regression:
    def fit(self, X, y):
        self.coef_ = scipy.linalg.solve(X,y)
        return self

    def predict(self, X):
        return X @ self.coef_
    
# read training data and test data
x_train, y_train, x_test, y_test, grid = get_data(2048-200, 200)

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

bandwidth = 5
kernel = RBF(length_scale = bandwidth)
model = GaussianProcessRegressor(kernel, alpha = 1e-10)

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Gaussian kernel Prediction error is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')

kernel = Matern(nu = 1, length_scale = 50.0)
model = GaussianProcessRegressor(kernel, alpha = 1e-10, normalize_y = False)

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 2 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')

kernel = Matern(nu = 1.5, length_scale = 50.0)
model = GaussianProcessRegressor(kernel, alpha = 1e-10, normalize_y = False)

start = time.time()
model.fit(x_train_noise, y_train_noise)
pred = model.predict(x_test_noise)
end = time.time()
e = np.mean(np.linalg.norm(pred - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))

print(f'Matern kernel Prediction error with nu = 3 is {e:.2e} \n')
print(f'Computation time is {end-start:.2f}')