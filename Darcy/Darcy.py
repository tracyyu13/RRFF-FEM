import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from scipy import io
from scipy.spatial import Delaunay
import time
from matplotlib import cm
from RF import *
from tqdm import tqdm

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

# Save data to text file
with open('Darcy_data_noise.txt','w') as f:
    np.savetxt(f, x_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_test, delimiter=',')

# Visualization
idx = 1

# training input
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(x_train_noise[idx].reshape(29,29), interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

# training output
im2 = ax[1].imshow(y_train_noise[idx].reshape(29,29), interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_training_data.png', bbox_inches = 'tight')

num = 20

######################## Gaussian random feature
# number of features
N = 10000
# scaling parameter gamma
gamma = 1e-5
alpha1=1e-10

alphas = np.array([10**-p for p in range(1,11)])
alphas = np.insert(alphas, 0, np.array([0.5,0.25]))
errors1 = np.zeros((num,))
times1 = np.zeros((num,))
errors_reg1 = np.zeros((num,alphas.shape[0]))
times_reg1 = np.zeros((num,alphas.shape[0]))
preds_reg1 = []
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
    
    for j in range(alphas.shape[0]):
        # regularization
        alpha = alphas[j]
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
        errors_reg1[i,j] = e
        times_reg1[i,j] = end-start
        if i == num-1:
            preds_reg1.append(pred2)

print(f'Average test error of Gaussian random feature model over {num} trials is {np.mean(errors1):.2e}.')
print(f'Average clock time is {np.mean(times1):.2f} seconds \n')

avg_err1 = np.mean(errors_reg1, axis=0)
avg_times1 = np.mean(times_reg1, axis=0)
std_err1 = np.std(errors_reg1, axis=0)
i = np.argmin(avg_err1)
alpha_gaussian = alphas[i]
print(f'Average test error of regularized Gaussian random feature model with \u03b1 = {alpha_gaussian} is {avg_err1[i]:.2e}.')
print(f'Average clock time is {avg_times1[i]:.2f} seconds \n')

# Save data to text file
with open('Darcy_test_pred_Gaussian.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg1[i], delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(29,29)
prediction1 = pred1[idx].reshape(29,29)
prediction2 = preds_reg1[i][idx].reshape(29,29)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_test_pred_Gaussian.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_error_Gaussian.png', bbox_inches = 'tight')

# Figure 3: alpha vs test error
fig, ax = plt.subplots()
ax.errorbar(np.log10(alphas), avg_err1, yerr=std_err1, color = cm.coolwarm(0.0), capsize=4)
ax.set_xlabel('$log_{10}(\u03b1)$', size= 25)
ax.set_ylabel('Test error', size= 25)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Darcy_alpha_error_Gaussian.png', bbox_inches = 'tight')

#plt.yticks(np.arange(4.2e-2,6.2e-2, 0.4e-2), fontsize=20)

######################## Student random feature nu=2
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv1 = 2
alpha1=1e-10

errors2 = np.zeros((num,))
times2 = np.zeros((num,))
errors_reg2 = np.zeros((num,alphas.shape[0]))
times_reg2 = np.zeros((num,alphas.shape[0]))
preds_reg2 = []
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

    for j in range(alphas.shape[0]):
        # regularization
        alpha = alphas[j]
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
        errors_reg2[i,j] = e
        times_reg2[i,j] = end-start
        if i == num-1:
            preds_reg2.append(pred2)

print(f'Average test error of Student random feature model with \u03BD = {nv1} over {num} trials is {np.mean(errors2):.2e}.')
print(f'Average clock time is {np.mean(times2):.2f} seconds \n')

avg_err2 = np.mean(errors_reg2, axis=0)
avg_times2 = np.mean(times_reg2, axis=0)
std_err2 = np.std(errors_reg2, axis=0)
i = np.argmin(avg_err2)
alpha_student2 = alphas[i]
print(f'Average test error of regularized Student random feature model with \u03BD = {nv1} and \u03b1 = {alpha_student2} is {avg_err2[i]:.2e}.')
print(f'Average clock time is {avg_times2[i]:.2f} seconds \n')

# Save data to text file
with open('Darcy_test_pred_Student2.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg2[i], delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(29,29)
prediction1 = pred1[idx].reshape(29,29)
prediction2 = preds_reg2[i][idx].reshape(29,29)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im3, cax=cax3)
cbar2.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_test_pred_Student2.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv2 = 3
alpha1=1e-10

errors3 = np.zeros((num,))
times3 = np.zeros((num,))
errors_reg3 = np.zeros((num,alphas.shape[0]))
times_reg3 = np.zeros((num,alphas.shape[0]))
preds_reg3 = []
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

    for j in range(alphas.shape[0]):
        # regularization
        alpha = alphas[j]
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
        errors_reg3[i,j] = e
        times_reg3[i,j] = end-start
        if i == num-1:
            preds_reg3.append(pred2)

print(f'Average test error of Student random feature model with \u03BD = {nv2} over {num} trials is {np.mean(errors3):.2e}.')
print(f'Average clock time is {np.mean(times3):.2f} seconds \n')

avg_err3 = np.mean(errors_reg3, axis=0)
avg_times3 = np.mean(times_reg3, axis=0)
std_err3 = np.std(errors_reg3, axis=0)
i = np.argmin(avg_err3)
alpha_student3 = alphas[i]
print(f'Average test error of regularized Student random feature model with \u03BD = {nv2} and \u03b1 = {alpha_student3} is {avg_err3[i]:.2e}.')
print(f'Average clock time is {avg_times3[i]:.2f} seconds \n')

# Save data to text file
with open('Darcy_test_pred_Student3.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg3[i], delimiter=',')

# Visualize the result
idx = 20

norm = np.linalg.norm(y_test[idx])
test = y_test[idx].reshape(29,29)
prediction1 = pred1[idx].reshape(29,29)
prediction2 = preds_reg3[i][idx].reshape(29,29)
error1 = (prediction1 - test)/norm
error2 = (prediction2 - test)/norm

# get min and max values for colormap
minmin = np.min([np.min(test), np.min(prediction1), np.min(prediction2)])
maxmax = np.max([np.max(test), np.max(prediction1), np.max(prediction2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(test, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(prediction1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(prediction2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_test_pred_Student3.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_error_Student3.png', bbox_inches = 'tight')

# Figure 3: alpha vs test error
fig, ax = plt.subplots()
ax.errorbar(np.log10(alphas), avg_err2, yerr=std_err2, label = f"Student RFFs: \u03BD = {nv1}", color = cm.coolwarm(0.0), capsize=4)
ax.errorbar(np.log10(alphas), avg_err3, yerr=std_err3, label = f"Student RFFs: \u03BD = {nv2}", color = 'red', capsize=4)
ax.set_xlabel('$log_{10}(\u03b1)$', size= 25)
ax.set_ylabel('Test error', size= 25)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20) 
# Display plot
plt.savefig('Darcy_alpha_error_Student.png', bbox_inches = 'tight')

#plt.yticks(np.arange(0.8e-1,1.3e-1, 0.1e-1), fontsize=20)

# Save data to text file
with open('Darcy_alpha_error.txt','w') as f:
    np.savetxt(f, alphas, delimiter=',')
    f.write('\n')
    np.savetxt(f, avg_err1, delimiter=',')
    f.write('\n')
    np.savetxt(f, std_err1, delimiter=',')
    f.write('\n')
    np.savetxt(f, avg_err2, delimiter=',')
    f.write('\n')
    np.savetxt(f, std_err2, delimiter=',')
    f.write('\n')
    np.savetxt(f, avg_err3, delimiter=',')
    f.write('\n')
    np.savetxt(f, std_err3, delimiter=',')

######################## Recovery Map ########################

import dolfinx
import ufl
from mpi4py import MPI
import basix

# save some training samples for testing combined f_hat and recovery map
S = 200 # number of training samples to save
M = x_train_noise.shape[0]-S

# split grid
mask = [i % 3 != 2 for i in range(x_train_noise.shape[1])]
grid_split = grid[mask,:]
x_train_split = x_train_noise[0:M,mask]
y_train_split = y_train_noise[0:M,mask]
x_test_split = x_test_noise[:,mask]
y_test_split = y_test[:,mask]

print('f_hat Training Input Shape: ' + str(x_train_split.shape))
print('f_hat Training Output Shape: ' + str(y_train_split.shape))
print('f_hat Testing Input Shape: ' + str(x_test_split.shape))
print('f_hat Testing Output Shape: ' + str(y_test_split.shape) + '\n')

# Save data to text file
with open('Darcy_data_split.txt','w') as f:
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

cmap = mesh.geometry.cmap
dofmap = mesh.geometry.dofmap.array.reshape(-1,3)
lagrange = basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, degree, basix.LagrangeVariant.equispaced)

######################## Gaussian random feature
# number of features
N = 10000
# scaling parameter gamma
gamma = 1e-5
alpha1=1e-10

errors1 = np.zeros((num,))
errors_reg1 = np.zeros((num,))
errors_recovery1 = np.zeros((num,))
errors_recovery_reg1 = np.zeros((num,))
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
    alpha = alpha_gaussian
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

    u_split = x_train_noise[M:,mask]
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

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery1[i] = e1
    errors_recovery_reg1[i] = e2

print(f'Average test error of Gaussian random feature model on split grid over {num} trials is {np.mean(errors1):.2e}.')
print(f'Average test error of regularized Gaussian random feature model on split grid with \u03b1 = {alpha_gaussian} is {np.mean(errors_reg1):.2e}. \n')

print(f'Average test error of f_hat (Gaussian RFs) & recovery map over {num} trials is {np.mean(errors_recovery1):.2e}. ')
print(f'Average test error of f_hat (regularized Gaussian RFs: \u03b1 = {alpha_gaussian}) & recovery map is {np.mean(errors_recovery_reg1):.2e}. \n')

# Save data to text file
with open('Darcy_recovery_Gaussian.txt','w') as f:
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

norm = np.linalg.norm(y_train[M + idx])
true = y_train[M + idx].reshape(29,29)
interpolant1 = v_out1[idx].reshape(29,29)
interpolant2 = v_out2[idx].reshape(29,29)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_interpolant_Gaussian.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_error_Gaussian.png', bbox_inches = 'tight')

######################## Student random feature nu=2
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv1 = 2
alpha1=1e-10

errors2 = np.zeros((num,))
errors_reg2 = np.zeros((num,))
errors_recovery2 = np.zeros((num,))
errors_recovery_reg2 = np.zeros((num,))
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
    alpha = alpha_student2
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

    u_split = x_train_noise[M:,mask]
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

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery2[i] = e1
    errors_recovery_reg2[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv1} on split grid over {num} trials is {np.mean(errors2):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv1} and \u03b1 = {alpha_student2} on split grid is {np.mean(errors_reg2):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv1}) & recovery map over {num} trials is {np.mean(errors_recovery2):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv1} and \u03b1 = {alpha_student2}) & recovery map is {np.mean(errors_recovery_reg2):.2e}. \n')

# Save data to text file
with open('Darcy_recovery_Student2.txt','w') as f:
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

norm = np.linalg.norm(y_train[M + idx])
true = y_train[M + idx].reshape(29,29)
interpolant1 = v_out1[idx].reshape(29,29)
interpolant2 = v_out2[idx].reshape(29,29)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_interpolant_Student2.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv2 = 3
alpha1=1e-10

errors3 = np.zeros((num,))
errors_reg3 = np.zeros((num,))
errors_recovery3 = np.zeros((num,))
errors_recovery_reg3 = np.zeros((num,))
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
    alpha = alpha_student3
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

    u_split = x_train_noise[M:,mask]
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

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery3[i] = e1
    errors_recovery_reg3[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv2} on split grid over {num} trials is {np.mean(errors3):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv2} and \u03b1 = {alpha_student3} on split grid is {np.mean(errors_reg3):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv2}) & recovery map over {num} trials is {np.mean(errors_recovery3):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv2} and \u03b1 = {alpha_student3}) & recovery map is {np.mean(errors_recovery_reg3):.2e}. \n')

# Save data to text file
with open('Darcy_recovery_Student3.txt','w') as f:
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

norm = np.linalg.norm(y_train[M + idx])
true = y_train[M + idx].reshape(29,29)
interpolant1 = v_out1[idx].reshape(29,29)
interpolant2 = v_out2[idx].reshape(29,29)
error1 = (interpolant1 - true)/norm
error2 = (interpolant2 - true)/norm

# get min and max values for colormap
minmin = np.min([np.min(true), np.min(interpolant1), np.min(interpolant2)])
maxmax = np.max([np.max(true), np.max(interpolant1), np.max(interpolant2)])

# test + predictions
fig, ax = plt.subplots(1,3, figsize=(15,5))
im1 = ax[0].imshow(true, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=15)

im2 = ax[1].imshow(interpolant1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=15)

im3 = ax[2].imshow(interpolant2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[2].set_xticks(np.linspace(0,28,5))
ax[2].set_yticks(np.linspace(0,28,5))
ax[2].set_xticklabels(np.linspace(0,1,5))
ax[2].set_yticklabels(np.linspace(0,1,5))
ax[2].tick_params(axis='x', labelsize=15)
ax[2].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[2])
cax3 = divider.append_axes("right", size="5%", pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.ax.tick_params(labelsize=15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_interpolant_Student3.png', bbox_inches = 'tight')

# get min and max values for colormap
minmin = np.min([np.min(error1), np.min(error2)])
maxmax = np.max([np.max(error1), np.max(error2)])

# pointwise errors
fig, ax = plt.subplots(1,2, figsize=(10,5))
im1 = ax[0].imshow(error1, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[0].set_xticks(np.linspace(0,28,5))
ax[0].set_yticks(np.linspace(0,28,5))
ax[0].set_xticklabels(np.linspace(0,1,5))
ax[0].set_yticklabels(np.linspace(0,1,5))
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[0])
cax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1)
cbar1.formatter.set_powerlimits((0, 0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)

im2 = ax[1].imshow(error2, vmin=minmin, vmax=maxmax, interpolation='bilinear', cmap= "coolwarm", origin= 'lower')
ax[1].set_xticks(np.linspace(0,28,5))
ax[1].set_yticks(np.linspace(0,28,5))
ax[1].set_xticklabels(np.linspace(0,1,5))
ax[1].set_yticklabels(np.linspace(0,1,5))
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2)
cbar2.formatter.set_powerlimits((0, 0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.grid(visible=False)
plt.tight_layout()
plt.savefig('Darcy_recovery_error_Student3.png', bbox_inches = 'tight')