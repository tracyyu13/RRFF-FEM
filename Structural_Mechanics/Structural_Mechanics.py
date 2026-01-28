import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import scipy
from scipy import io
from scipy.spatial import Delaunay
from matplotlib import cm
from RF import *
from tqdm import tqdm

# uses scipy.linalg.solve
class Cholesky_Regression:
    def fit(self, X, y):
        self.coef_ = scipy.linalg.solve(X,y)
        return self

    def predict(self, X):
        return X @ self.coef_

# Get Data

Inputs = np.load('StructuralMechanics_inputs.npy')
Outputs = np.load('StructuralMechanics_outputs.npy')
print('Loaded data\n')

xgrid = np.linspace(0,1,41, dtype=np.float32)
grids = []
grids.append(xgrid)
grids.append(xgrid)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

print('Formed grid \n')

# transpose it and then it is easy to reshape

Inputs = Inputs[:,0,:]
Outputs = Outputs.transpose((2,1,0))

# flatten it

Inputs_fl = Inputs.T.reshape(len(Inputs.T), 41)
Outputs_fl = Outputs.reshape(40000, 41*41)

# train_test split
Ntrain = 20000
x_train = Inputs_fl[:Ntrain]
x_test = Inputs_fl[Ntrain:]

y_train = Outputs_fl[:Ntrain]
y_test = Outputs_fl[Ntrain:]

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

# Save data to text file
with open('Structural_data_noise.txt','w') as f:
    np.savetxt(f, x_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_test, delimiter=',')

# Visualize one training inputs
idx = 1

# one example of training input
plt.figure()
plt.plot(xgrid, x_train_noise[idx], color = cm.coolwarm(0.0), linewidth=2)
plt.xlabel(r'$x$', size=15)
plt.ylabel(r'$u(x)$', size=15)
plt.xticks([0,1])
plt.yticks([-250,0,250,500])
plt.title('Training Input', fontsize=15)
plt.savefig('Structural_input.png', bbox_inches = 'tight')

# training output
fig, ax1 = plt.subplots(1,1, figsize=(15,5))
im = ax1.imshow(y_train_noise[idx].reshape(41,41), interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax1.set_xticks(np.linspace(0,40,5))
ax1.set_yticks(np.linspace(0,40,5))
ax1.set_xticklabels(np.linspace(0,1,5))
ax1.set_yticklabels(np.linspace(0,1,5))
ax1.set_title('Training Output', fontsize=15)
plt.grid(visible=False)
cb = fig.colorbar(im, ax=ax1, pad=0.008)
plt.savefig('Structural_output.png', bbox_inches = 'tight')

num = 20

######################## Gaussian random feature
# number of features
N = 25000
# scaling parameter gamma
gamma =  1e-6
alpha1=1e-10

errors1 = np.zeros(num)
times1 = np.zeros(num)
errors_reg1 = np.zeros(num)
times_reg1 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W, random_offset, x_train_RF, x_test_RF = RF_Gaussian(gamma, N, x_train_noise, x_test_noise)
    X = x_train_RF.T @ x_train_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_RF.T @ y_train_noise

    # train a linear regression model
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X1,y)
    pred1 = model.predict(x_test_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors1[i] = e
    times1[i] = end-start

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W[:,i])**p) for i in range(0,W.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X2, y)
    pred2 = model.predict(x_test_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors_reg1[i] = e
    times_reg1[i] = end-start

print(f'Average test error of Gaussian random feature model over {num} trials is {np.mean(errors1):.2e}.')
print(f'Average clock time is {np.mean(times1):.2f} seconds \n')

print(f'Average test error of regularized Gaussian random feature model with \u03b1 = {alpha} is {np.mean(errors_reg1):.2e}.')
print(f'Average clock time is {np.mean(times_reg1):.2f} seconds \n')

# Save data to text file
with open('Structural_test_pred_Gaussian.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred2, delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(41,41)
prediction1 = pred1[idx].reshape(41,41)
prediction2 = pred2[idx].reshape(41,41)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF Prediction', fontsize=15)
ax[2].set_title(f'RRFF Prediction: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_test_pred_Gaussian.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('RFF Pointwise Error', fontsize=15)
ax[1].set_title(f'RRFF Pointwise Error: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_error_Gaussian.png', bbox_inches = 'tight')

######################## Student random feature nu=2
# number of features
N = 25000
# scaling parameter gamma
gamma = 1e-3
nv1 = 2
alpha1=1e-10

errors2 = np.zeros(num)
times2 = np.zeros(num)
errors_reg2 = np.zeros(num)
times_reg2 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W, random_offset, x_train_RF, x_test_RF = RF_student(nv1, gamma, N, x_train_noise, x_test_noise)
    X = x_train_RF.T @ x_train_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_RF.T @ y_train_noise

    # train a linear regression model
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X1,y)
    pred1 = model.predict(x_test_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors2[i] = e
    times2[i] = end-start

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W[:,i])**p) for i in range(0,W.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X2, y)
    pred2 = model.predict(x_test_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors_reg2[i] = e
    times_reg2[i] = end-start

print(f'Average test error of Student random feature model with \u03BD = {nv1} over {num} trials is {np.mean(errors2):.2e}.')
print(f'Average clock time is {np.mean(times2):.2f} seconds \n')

print(f'Average test error of regularized Student random feature model with \u03BD = {nv1} and \u03b1 = {alpha} is {np.mean(errors_reg2):.2e}.')
print(f'Average clock time is {np.mean(times_reg2):.2f} seconds \n')

# Save data to text file
with open('Structural_test_pred_Student2.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred2, delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(41,41)
prediction1 = pred1[idx].reshape(41,41)
prediction2 = pred2[idx].reshape(41,41)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF Prediction', fontsize=15)
ax[2].set_title(f'RRFF Prediction: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_test_pred_Student2.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('RFF Pointwise Error', fontsize=15)
ax[1].set_title(f'RRFF Pointwise Error: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 25000
# scaling parameter gamma
gamma = 1e-3
nv2 = 3
alpha1=1e-10

errors3 = np.zeros(num)
times3 = np.zeros(num)
errors_reg3 = np.zeros(num)
times_reg3 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W, random_offset, x_train_RF, x_test_RF = RF_student(nv2, gamma, N, x_train_noise, x_test_noise)
    X = x_train_RF.T @ x_train_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_RF.T @ y_train_noise

    # train a linear regression model
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X1,y)
    pred1 = model.predict(x_test_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors3[i] = e
    times3[i] = end-start

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W[:,i])**p) for i in range(0,W.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model = Cholesky_Regression()
    start = time.time()
    model.fit(X2, y)
    pred2 = model.predict(x_test_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1))
    errors_reg3[i] = e
    times_reg3[i] = end-start

print(f'Average test error of Student random feature model with \u03BD = {nv2} over {num} trials is {np.mean(errors3):.2e}.')
print(f'Average clock time is {np.mean(times3):.2f} seconds \n')

print(f'Average test error of regularized Student random feature model with \u03BD = {nv2} and \u03b1 = {alpha} is {np.mean(errors_reg3):.2e}.')
print(f'Average clock time is {np.mean(times_reg3):.2f} seconds \n')

# Save data to text file
with open('Structural_test_pred_Student3.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred2, delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(41,41)
prediction1 = pred1[idx].reshape(41,41)
prediction2 = pred2[idx].reshape(41,41)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF Prediction', fontsize=15)
ax[2].set_title(f'RRFF Prediction: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_test_pred_Student3.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('RFF Pointwise Error', fontsize=15)
ax[1].set_title(f'RRFF Pointwise Error: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_error_Student3.png', bbox_inches = 'tight')

######################## Recovery Map ########################

import dolfinx
import ufl
from mpi4py import MPI
import basix

# save some testing samples for testing combined f_hat and recovery map
S = 5000 # number of testing samples to save
M = x_test_noise.shape[0]-S 

# split grid
xmask = [i % 3 != 2 for i in range(x_test_noise.shape[1])]
xgrid_split = xgrid[xmask]

masks = []
masks.append(np.array(xmask))
masks.append(np.array(xmask))
mask = np.vstack([xx.ravel() for xx in np.meshgrid(*masks)]).T

for i in range(41*41):
    if mask[i,0] == False and mask[i,1] == True:
        mask[i,1] = False
    if mask[i,0] == True and mask[i,1] == False:
        mask[i,0] = False

grid_split = grid[mask]
grid_split = grid_split.reshape(-1,2)

x_train_split = x_train_noise[:,xmask]
x_test_split = x_test_noise[0:M,xmask]

y_train_split = y_train_noise[:,mask[:,0]]
y_test_split = y_test[0:M,mask[:,0]]

print('f_hat Training Input Shape: ' + str(x_train_split.shape))
print('f_hat Training Output Shape: ' + str(y_train_split.shape))
print('f_hat Testing Input Shape: ' + str(x_test_split.shape))
print('f_hat Testing Output Shape: ' + str(y_test_split.shape) + '\n')

# Save data to text file
with open('Structural_data_split.txt','w') as f:
    np.savetxt(f, grid, delimiter=',')
    f.write('\n')
    np.savetxt(f, grid_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_test_split, delimiter=',')

gdim = 3
shape = "triangle"
degree = 1

cell = ufl.Cell(shape, gdim)
element = ufl.VectorElement("Lagrange", cell, degree)
domain = ufl.Mesh(element)

nodes = grid_split
tri = Delaunay(nodes)
cells = tri.simplices
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, nodes, domain)

mesh_nodes = mesh.geometry.x

###### identify which cells the points are in
bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)

cells = []
# Find cells whose bounding-box collide with the the points
cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, grid)
# Choose one of the cells that contains the point
colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, grid)
for i, point in enumerate(grid):
    if len(colliding_cells.links(i)) > 0:
        cells.append(colliding_cells.links(i)[0])

# Pull cells back to reference element
cmap = mesh.geometry.cmap
dofmap = mesh.geometry.dofmap.array.reshape(-1,3)
lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced)

######################## Gaussian random feature
# number of features
N = 25000
# scaling parameter gamma
gamma =  1e-6
alpha1=1e-10

errors1 = np.zeros(num)
errors_reg1 = np.zeros(num)
errors_recovery1 = np.zeros(num)
errors_recovery_reg1 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W_split, b_split, x_train_split_RF, x_test_split_RF = RF_Gaussian(gamma, N, x_train_split, x_test_split)
    X = x_train_split_RF.T @ x_train_split_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_split_RF.T @ y_train_split

    # train a linear regression model
    model_f1 = Cholesky_Regression()
    start = time.time()
    model_f1.fit(X1,y)
    pred1 = model_f1.predict(x_test_split_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors1[i] = e

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W_split[:,i])**p) for i in range(0,W_split.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model_f2 = Cholesky_Regression()
    start = time.time()
    model_f2.fit(X2, y)
    pred2 = model_f2.predict(x_test_split_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors_reg1[i] = e

    u_split = x_test_noise[M:,xmask]
    u_split_RF = np.cos(u_split @ W_split + b_split) * (2.0 / N) ** 0.5
    v_split_pred1 = model_f1.predict(u_split_RF)
    v_split_pred2 = model_f2.predict(u_split_RF)

    v_out1 = np.zeros((S,grid.shape[0]))
    v_out2 = np.zeros((S,grid.shape[0]))
    # Pull cells back to reference element
    for k, (point, cell) in enumerate(zip(grid, cells)):
        geom_dofs = dofmap[cell]
        cell_coords = mesh_nodes[geom_dofs]
        # Find reference element
        ref_x = cmap.pull_back(np.array([point]), cell_coords)
        # Evaluate basis functions at reference element
        tab = lagrange.tabulate(0, ref_x)[0][0]

        indices = np.zeros((3,),dtype=int)
        for j in range(3):
            x_coord = cell_coords[j,0]
            y_coord = cell_coords[j,1]
            ind_x = np.where(grid_split[:,0]==x_coord)[0]
            ind_y = np.where(grid_split[:,1]==y_coord)[0]
            indices[j] = np.intersect1d(ind_x,ind_y)[0]

        v_vals1 = np.array([v_in[indices] for v_in in v_split_pred1])
        v_vals2 = np.array([v_in[indices] for v_in in v_split_pred2])
        v_out1[:,k] = (v_vals1 @ tab)[:,0]
        v_out2[:,k] = (v_vals2 @ tab)[:,0]

    e1 = np.mean(np.linalg.norm(v_out1 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    errors_recovery1[i] = e1
    errors_recovery_reg1[i] = e2

print(f'Average test error of Gaussian random feature model on split grid over {num} trials is {np.mean(errors1):.2e}.')
print(f'Average test error of regularized Gaussian random feature model on split grid with \u03b1 = {alpha} is {np.mean(errors_reg1):.2e}. \n')

print(f'Average test error of f_hat (Gaussian RFs) & recovery map over {num} trials is {np.mean(errors_recovery1):.2e}. ')
print(f'Average test error of f_hat (regularized Gaussian RFs: \u03b1 = {alpha}) & recovery map is {np.mean(errors_recovery_reg1):.2e}. \n')

# Save data to text file
with open('Structural_recovery_Gaussian.txt','w') as f:
    np.savetxt(f, W_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, b_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out1, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out2, delimiter=',')

# Visualize the result
idx = 120

norm = np.linalg.norm(y_test[M + idx])
true = y_test[M + idx].reshape(41,41)
interpolant1 = v_out1[idx].reshape(41,41)
interpolant2 = v_out2[idx].reshape(41,41)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF + FEM Prediction', fontsize=15)
ax[2].set_title(f'RRFF + FEM Prediction: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_interpolant_Gaussian.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('Pointwise Error - RFF + FEM', fontsize=15)
ax[1].set_title(f'Pointwise Error - RRFF + FEM: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_error_Gaussian.png', bbox_inches = 'tight')

######################## Student random feature nu=2
# number of features
N = 25000
# scaling parameter gamma
gamma = 1e-3
nv1 = 2
alpha1=1e-10

errors2 = np.zeros(num)
errors_reg2 = np.zeros(num)
errors_recovery2 = np.zeros(num)
errors_recovery_reg2 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W_split, b_split, x_train_split_RF, x_test_split_RF = RF_student(nv1, gamma, N, x_train_split, x_test_split)
    X = x_train_split_RF.T @ x_train_split_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_split_RF.T @ y_train_split

    # train a linear regression model
    model_f1 = Cholesky_Regression()
    start = time.time()
    model_f1.fit(X1,y)
    pred1 = model_f1.predict(x_test_split_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors2[i] = e

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W_split[:,i])**p) for i in range(0,W_split.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model_f2 = Cholesky_Regression()
    start = time.time()
    model_f2.fit(X2, y)
    pred2 = model_f2.predict(x_test_split_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors_reg2[i] = e

    u_split = x_test_noise[M:,xmask]
    u_split_RF = np.cos(u_split @ W_split + b_split) * (2.0 / N) ** 0.5
    v_split_pred1 = model_f1.predict(u_split_RF)
    v_split_pred2 = model_f2.predict(u_split_RF)

    v_out1 = np.zeros((S,grid.shape[0]))
    v_out2 = np.zeros((S,grid.shape[0]))
    # Pull cells back to reference element
    for k, (point, cell) in enumerate(zip(grid, cells)):
        geom_dofs = dofmap[cell]
        cell_coords = mesh_nodes[geom_dofs]
        # Find reference element
        ref_x = cmap.pull_back(np.array([point]), cell_coords)
        # Evaluate basis functions at reference element
        tab = lagrange.tabulate(0, ref_x)[0][0]

        indices = np.zeros((3,),dtype=int)
        for j in range(3):
            x_coord = cell_coords[j,0]
            y_coord = cell_coords[j,1]
            ind_x = np.where(grid_split[:,0]==x_coord)[0]
            ind_y = np.where(grid_split[:,1]==y_coord)[0]
            indices[j] = np.intersect1d(ind_x,ind_y)[0]

        v_vals1 = np.array([v_in[indices] for v_in in v_split_pred1])
        v_vals2 = np.array([v_in[indices] for v_in in v_split_pred2])
        v_out1[:,k] = (v_vals1 @ tab)[:,0]
        v_out2[:,k] = (v_vals2 @ tab)[:,0]

    e1 = np.mean(np.linalg.norm(v_out1 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    errors_recovery2[i] = e1
    errors_recovery_reg2[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv1} on split grid over {num} trials is {np.mean(errors2):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv1} and \u03b1 = {alpha} on split grid is {np.mean(errors_reg2):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv1}) & recovery map over {num} trials is {np.mean(errors_recovery2):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv1} and \u03b1 = {alpha}) & recovery map is {np.mean(errors_recovery_reg2):.2e}. \n')

# Save data to text file
with open('Structural_recovery_Student2.txt','w') as f:
    np.savetxt(f, W_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, b_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out1, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out2, delimiter=',')

# Visualize the result
idx = 120

norm = np.linalg.norm(y_test[M + idx])
true = y_test[M + idx].reshape(41,41)
interpolant1 = v_out1[idx].reshape(41,41)
interpolant2 = v_out2[idx].reshape(41,41)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF + FEM Prediction', fontsize=15)
ax[2].set_title(f'RRFF + FEM Prediction: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_interpolant_Student2.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('Pointwise Error - RFF + FEM', fontsize=15)
ax[1].set_title(f'Pointwise Error - RRFF + FEM: \u03b1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 25000
# scaling parameter gamma
gamma = 1e-3
nv2 = 3
alpha1=1e-10

errors3 = np.zeros(num)
errors_reg3 = np.zeros(num)
errors_recovery3 = np.zeros(num)
errors_recovery_reg3 = np.zeros(num)
for i in tqdm(range(num)):
    # generate random feature matrix
    W_split, b_split, x_train_split_RF, x_test_split_RF = RF_student(nv2, gamma, N, x_train_split, x_test_split)
    X = x_train_split_RF.T @ x_train_split_RF
    X1 = X + alpha1*np.eye(X.shape[0])
    y = x_train_split_RF.T @ y_train_split

    # train a linear regression model
    model_f1 = Cholesky_Regression()
    start = time.time()
    model_f1.fit(X1,y)
    pred1 = model_f1.predict(x_test_split_RF)
    end = time.time()
    # relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred1 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors3[i] = e

    # regularization
    alpha = 0.5
    p=2
    b = np.array([(np.linalg.norm(W_split[:,i])**p) for i in range(0,W_split.shape[1])])
    b = alpha*b/np.max(b)
    b[b<alpha1] = alpha1
    B = np.diag(b)

    # train a linear regression model
    X2 = X + B
    model_f2 = Cholesky_Regression()
    start = time.time()
    model_f2.fit(X2, y)
    pred2 = model_f2.predict(x_test_split_RF)
    end = time.time()
    # report relative prediction error and clock time
    e = np.mean(np.linalg.norm(pred2 - y_test_split, axis = -1)/np.linalg.norm(y_test_split, axis = -1))
    errors_reg3[i] = e

    u_split = x_test_noise[M:,xmask]
    u_split_RF = np.cos(u_split @ W_split + b_split) * (2.0 / N) ** 0.5
    v_split_pred1 = model_f1.predict(u_split_RF)
    v_split_pred2 = model_f2.predict(u_split_RF)

    v_out1 = np.zeros((S,grid.shape[0]))
    v_out2 = np.zeros((S,grid.shape[0]))
    for k, (point, cell) in enumerate(zip(grid, cells)):
        geom_dofs = dofmap[cell]
        cell_coords = mesh_nodes[geom_dofs]
        # Find reference element
        ref_x = cmap.pull_back(np.array([point]), cell_coords)
        # Evaluate basis functions at reference element
        tab = lagrange.tabulate(0, ref_x)[0][0]

        indices = np.zeros((3,),dtype=int)
        for j in range(3):
            x_coord = cell_coords[j,0]
            y_coord = cell_coords[j,1]
            ind_x = np.where(grid_split[:,0]==x_coord)[0]
            ind_y = np.where(grid_split[:,1]==y_coord)[0]
            indices[j] = np.intersect1d(ind_x,ind_y)[0]

        v_vals1 = np.array([v_in[indices] for v_in in v_split_pred1])
        v_vals2 = np.array([v_in[indices] for v_in in v_split_pred2])
        v_out1[:,k] = (v_vals1 @ tab)[:,0]
        v_out2[:,k] = (v_vals2 @ tab)[:,0]

    e1 = np.mean(np.linalg.norm(v_out1 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_test[M:,:], axis = -1)/np.linalg.norm(y_test[M:,:], axis = -1))
    errors_recovery3[i] = e1
    errors_recovery_reg3[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv2} on split grid over {num} trials is {np.mean(errors3):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv2} and \u03b1 = {alpha} on split grid is {np.mean(errors_reg3):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv2}) & recovery map over {num} trials is {np.mean(errors_recovery3):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv2} and \u03b1 = {alpha}) & recovery map is {np.mean(errors_recovery_reg3):.2e}. \n')

# Save data to text file
with open('Structural_recovery_Student3.txt','w') as f:
    np.savetxt(f, W_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, b_split, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_split_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out1, delimiter=',')
    f.write('\n')
    np.savetxt(f, v_out2, delimiter=',')

# Visualize the result
idx = 120

norm = np.linalg.norm(y_test[M + idx])
true = y_test[M + idx].reshape(41,41)
interpolant1 = v_out1[idx].reshape(41,41)
interpolant2 = v_out2[idx].reshape(41,41)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,40,5))
ax[2].set_yticks(np.linspace(0,40,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im3, cax=cax3)

ax[0].set_title('Test Example', fontsize=15)
ax[1].set_title('RFF + FEM Prediction', fontsize=15)
ax[2].set_title(f'RRFF + FEM Prediction: \u03B1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_interpolant_Student3.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,40,5))
ax[0].set_yticks(np.linspace(0,40,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1).formatter.set_powerlimits((0, 0))

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,40,5))
ax[1].set_yticks(np.linspace(0,40,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2).formatter.set_powerlimits((0, 0))

ax[0].set_title('Pointwise Error - RFF + FEM', fontsize=15)
ax[1].set_title(f'Pointwise Error - RRFF + FEM: \u03B1 = {alpha}', fontsize=15)
plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Structural_recovery_error_Student3.png', bbox_inches = 'tight')