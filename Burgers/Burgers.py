import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from scipy import io
from matplotlib import cm
from RF import *
from tqdm import tqdm

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

# Save data to text file
with open('Burger_data_noise.txt','w') as f:
    np.savetxt(f, x_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train_noise, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_train, delimiter=',')
    f.write('\n')
    np.savetxt(f, y_test, delimiter=',')

# Visualize one training inputs
idx = 1

# one example of training input
plt.figure()
plt.plot(grid, x_train_noise[idx], color = cm.coolwarm(0.0), linewidth=2)
plt.xlabel(r'$x$', size=25)
plt.ylabel(r'$u(x) = w(x,0)$', size=25)
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.savefig('Burger_input.png', bbox_inches = 'tight')

# one example of training output
plt.figure()
plt.plot(grid, y_train_noise[idx], color = cm.coolwarm(0.0), linewidth=2)
plt.xlabel(r'$x$', size=25)
plt.ylabel(r'$v(x) = w(x,1)$', size=25)
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.savefig('Burger_output.png', bbox_inches = 'tight')

num = 20

######################## Gaussian random feature
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
alpha1=1e-10

alphas = np.array([10**-p for p in range(1,11)])
alphas = np.insert(alphas, 0, np.array([1,0.5,0.25]))
errors1 = np.zeros(num)
times1 = np.zeros(num)
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
with open('Burger_test_pred_Gaussian.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg1[i], delimiter=',')

# visualize the result for Gaussian RFs
idx = 50

# Figure 1: test + prediction
plt.figure()
plt.plot(grid, y_test[idx],  label = "Test Example",  linewidth=2.0, color='dimgrey')
plt.plot(grid, pred1[idx],  label = "RFF Prediction", linestyle='--',dashes=(4, 4), color = 'red',  linewidth=1.5)
plt.plot(grid, preds_reg1[i][idx],  label = f"RRFF Prediction: \u03b1 = {alpha_gaussian}", linestyle='--',dashes=(4, 4), color = cm.coolwarm(0.0),  linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.legend(fontsize='x-large',loc="upper right")
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.savefig('Burger_test_pred_Gaussian.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - test)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_test[idx])
ax.plot(grid, (pred1[idx]-y_test[idx])/norm, label="RFF", linewidth=1.5, color='red')
ax.plot(grid, (preds_reg1[i][idx]-y_test[idx])/norm, label = f"RRFF: \u03b1 = {alpha_gaussian}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Burger_error_Gaussian.png', bbox_inches = 'tight')

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
plt.savefig('Burger_alpha_error_Gaussian.png', bbox_inches = 'tight')

######################## Student random feature nu=2
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv1 = 2
alpha1=1e-10

alphas = np.array([10**-p for p in range(1,11)])
alphas = np.insert(alphas, 0, np.array([5,1,0.5,0.25]))
errors2 = np.zeros(num)
times2 = np.zeros(num)
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
with open('Burger_test_pred_Student2.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg2[i], delimiter=',')

# visualize the result for Student RFs with nu=2
idx = 50

# Figure 1: test + prediction
plt.figure()
plt.plot(grid, y_test[idx],  label = "Test Example",  linewidth=2.0, color='dimgrey')
plt.plot(grid, pred1[idx],  label = "RFF Prediction", linestyle='--',dashes=(4, 4), color = 'red',  linewidth=1.5)
plt.plot(grid, preds_reg2[i][idx],  label = f"RRFF Prediction: \u03b1 = {alpha_student2}", linestyle='--',dashes=(4, 4), color = cm.coolwarm(0.0),  linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.legend(fontsize='x-large',loc="upper right")
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.savefig('Burger_test_pred_Student2.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - test)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_test[idx])
ax.plot(grid, (pred1[idx]-y_test[idx])/norm, label="RFF", linewidth=1.5, color='red')
ax.plot(grid, (preds_reg2[i][idx]-y_test[idx])/norm, label = f"RRFF: \u03b1 = {alpha_student2}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Burger_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
nv2 = 3
alpha1=1e-10

errors3 = np.zeros(num)
times3 = np.zeros(num)
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
with open('Burger_test_pred_Student3.txt','w') as f:
    np.savetxt(f, W, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_train_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, x_test_RF, delimiter=',')
    f.write('\n')
    np.savetxt(f, pred1, delimiter=',')
    f.write('\n')
    np.savetxt(f, preds_reg3[i], delimiter=',')

# visualize the result for Student RFs with nu=3
idx = 50

# Figure 1: test + prediction
plt.figure()
plt.plot(grid, y_test[idx],  label = "Test Example",  linewidth=2.0, color='dimgrey')
plt.plot(grid, pred1[idx],  label = "RFF Prediction", linestyle='--',dashes=(4, 4), color = 'red',  linewidth=1.5)
plt.plot(grid, preds_reg3[i][idx],  label = f"RRFF Prediction: \u03b1 = {alpha_student3}", linestyle='--',dashes=(4, 4), color = cm.coolwarm(0.0),  linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.legend(fontsize='x-large',loc="upper right")
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.savefig('Burger_test_pred_Student3.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - test)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_test[idx])
ax.plot(grid, (pred1[idx]-y_test[idx])/norm, label="RFF", linewidth=1.5, color='red')
ax.plot(grid, (preds_reg3[i][idx]-y_test[idx])/norm, label = f"RRFF: \u03b1 = {alpha_student3}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20) 
# Display plot
plt.savefig('Burger_error_Student3.png', bbox_inches = 'tight')

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
plt.savefig('Burger_alpha_error_Student.png', bbox_inches = 'tight')

# Save data to text file
with open('Burger_alpha_error.txt','w') as f:
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

# save some training samples for testing combined f_hat and recovery map
M = x_train_noise.shape[0]-200 # number of training samples to keep

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
with open('Burger_data_split.txt','w') as f:
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

######################## Gaussian random feature
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
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

    # linear interpolation
    v_out1 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred1])
    v_out2 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred2])

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery1[i] = e1
    errors_recovery_reg1[i] = e2

print(f'Average test error of Gaussian random feature model on split grid over {num} trials is {np.mean(errors1):.2e}.')
print(f'Average test error of regularized Gaussian random feature model on split grid with \u03b1 = {alpha_gaussian} is {np.mean(errors_reg1):.2e}. \n')

print(f'Average test error of f_hat (Gaussian RFs) & recovery map over {num} trials is {np.mean(errors_recovery1):.2e}. ')
print(f'Average test error of f_hat (regularized Gaussian RFs: \u03b1 = {alpha_gaussian}) & recovery map is {np.mean(errors_recovery_reg1):.2e}. \n')

# Save data to text file
with open('Burger_recovery_Gaussian.txt','w') as f:
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

idx = 120

# Figure: true v + predictions
plt.figure()
plt.plot(grid, y_train[idx + M],  label = "Test Example",  linewidth=1.5, color= 'dimgrey')
plt.plot(grid, v_out1[idx],  label = "RFF + FEM Prediction",  linestyle='--',dashes=(4, 4), color= 'red', linewidth=1.5)
plt.plot(grid, v_out2[idx],  label = f"RRFF + FEM Prediction: \u03B1 = {alpha_gaussian}",  linestyle='--',dashes=(4, 4), color= cm.coolwarm(0.0), linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.legend(fontsize='x-large',loc="upper right")
plt.savefig('Burger_recovery_interpolant_Gaussian.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - true)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_train[idx + M])
ax.plot(grid, (v_out1[idx]-y_train[idx + M])/norm, label="RFF + FEM", linewidth=1.5, color='red')
ax.plot(grid, (v_out2[idx]-y_train[idx + M])/norm, label = f"RRFF + FEM: \u03b1 = {alpha_gaussian}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Burger_recovery_error_Gaussian.png', bbox_inches = 'tight')

######################## Student random feature nu=2
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
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

    # linear interpolation
    v_out1 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred1])
    v_out2 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred2])

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery2[i] = e1
    errors_recovery_reg2[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv1} on split grid over {num} trials is {np.mean(errors2):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv1} and \u03b1 = {alpha_student2} on split grid is {np.mean(errors_reg2):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv1}) & recovery map over {num} trials is {np.mean(errors_recovery2):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv1} and \u03b1 = {alpha_student2}) & recovery map is {np.mean(errors_recovery_reg2):.2e}. \n')

# Save data to text file
with open('Burger_recovery_Student2.txt','w') as f:
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

idx = 120

# Figure: true v + predictions
plt.figure()
plt.plot(grid, y_train[idx + M],  label = "Test Example",  linewidth=1.5, color= 'dimgrey')
plt.plot(grid, v_out1[idx],  label = "RFF + FEM Prediction",  linestyle='--',dashes=(4, 4), color= 'red', linewidth=1.5)
plt.plot(grid, v_out2[idx],  label = f"RRFF + FEM Prediction: \u03B1 = {alpha_student2}",  linestyle='--',dashes=(4, 4), color= cm.coolwarm(0.0), linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.legend(fontsize='x-large',loc="upper right")
plt.savefig('Burger_recovery_interpolant_Student2.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - true)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_train[idx + M])
ax.plot(grid, (v_out1[idx]-y_train[idx + M])/norm, label="RFF + FEM", linewidth=1.5, color='red')
ax.plot(grid, (v_out2[idx]-y_train[idx + M])/norm, label = f"RRFF + FEM: \u03b1 = {alpha_student2}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Burger_recovery_error_Student2.png', bbox_inches = 'tight')

######################## Student random feature nu=3
# number of features
N = 10000
# scaling parameter gamma
gamma = 0.02
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

    # linear interpolation
    v_out1 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred1])
    v_out2 = np.array([np.interp(grid[:,0], grid_split[:,0], v_in) for v_in in v_split_pred2])

    e1 = np.mean(np.linalg.norm(v_out1 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    e2 = np.mean(np.linalg.norm(v_out2 - y_train[M:,:], axis = -1)/np.linalg.norm(y_train[M:,:], axis = -1))
    errors_recovery3[i] = e1
    errors_recovery_reg3[i] = e2

print(f'Average test error of Student random feature model with \u03BD = {nv2} on split grid over {num} trials is {np.mean(errors3):.2e}.')
print(f'Average test error of regularized Student random feature model with \u03BD = {nv2} and \u03b1 = {alpha_student3} on split grid is {np.mean(errors_reg3):.2e}.\n')

print(f'Average test error of f_hat (Student RFs: \u03BD = {nv2}) & recovery map over {num} trials is {np.mean(errors_recovery3):.2e}. ')
print(f'Average test error of f_hat (regularized Student RFs: \u03BD = {nv2} and \u03b1 = {alpha_student3}) & recovery map is {np.mean(errors_recovery_reg3):.2e}. \n')

# Save data to text file
with open('Burger_recovery_Student3.txt','w') as f:
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

idx = 120

# Figure: true v + predictions
plt.figure()
plt.plot(grid, y_train[idx + M],  label = "Test Example",  linewidth=1.5, color= 'dimgrey')
plt.plot(grid, v_out1[idx],  label = "RFF + FEM Prediction",  linestyle='--',dashes=(4, 4), color= 'red', linewidth=1.5)
plt.plot(grid, v_out2[idx],  label = f"RRFF + FEM Prediction: \u03B1 = {alpha_student3}",  linestyle='--',dashes=(4, 4), color= cm.coolwarm(0.0), linewidth=1.5)
plt.xlabel(r'$x$', size= 25)
plt.ylabel(r'$w(x,1)$', size= 25)
plt.xticks([0,1], fontsize=20)
plt.yticks([-1,0,1], fontsize=20)
plt.legend(fontsize='x-large',loc="upper right")
plt.savefig('Burger_recovery_interpolant_Student3.png', bbox_inches = 'tight')

# Figure 2: pointwise error (Prediction - true)
fig, ax = plt.subplots()
norm = np.linalg.norm(y_train[idx + M])
ax.plot(grid, (v_out1[idx]-y_train[idx + M])/norm, label="RFF + FEM", linewidth=1.5, color='red')
ax.plot(grid, (v_out2[idx]-y_train[idx + M])/norm, label = f"RRFF + FEM: \u03b1 = {alpha_student3}", linewidth=1.5, color=cm.coolwarm(0.0))
ax.set_xlabel(r'$x$', size= 25)
ax.set_ylabel('Pointwise error', size= 25)
ax.legend(fontsize='x-large',loc="lower left")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.linspace(0,1,2))
ax.set_xticklabels([0,1], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
# Increase font size of scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)
# Display plot
plt.savefig('Burger_recovery_error_Student3.png', bbox_inches = 'tight')