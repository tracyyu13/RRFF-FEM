# Original code from Chunyang Liao
# Source: https://github.com/liaochunyang/RandomFeatureOperatorLearning/tree/main/RF

#!/usr/bin/env python
# coding: utf-8

import numpy as np

def RF_Gaussian(gamma, N, x_train, x_test):
    """
    Generate Gaussian Random Features, which approximate Gaussian kernel \exp(-\gamma\|x\|_2^2)
    
    Inputs:
    gamma: (2*gamma)**0.5 is the variance of the Gaussian distribution
    N: number of random features
    x_train: training samples of shape m x d
    x_test: test samples of shape m' x d
    
    Outputs:
    A_train: Training Random Feature Map A_train
    A_test: Test random feature map A_test
    """
    # number of samples and dimension of features
    m,d = x_train.shape
    rng = np.random.default_rng()
    # random features
    Omega = (2.0 * gamma) ** 0.5*rng.normal(size = (d,N))
    
    # Random feature matrix A
    random_offset = rng.uniform(0, 2 * np.pi, size=(1,N))
    A_train = np.cos(x_train@Omega + random_offset)
    A_test = np.cos(x_test@Omega + random_offset)
    
    return Omega, random_offset, A_train * (2.0 / N) ** 0.5, A_test * (2.0 / N) ** 0.5


def student(nu, sigma, size):
    rng = np.random.default_rng()
    gaussian = rng.normal(loc=0, scale=sigma, size=size)
    chisquare = rng.chisquare(nu, size=(1,size[1]))
    
    return np.sqrt(nu/chisquare)*gaussian


def RF_student(nu, gamma, N, x_train, x_test):
    """
    Generate Student Random Features, which approximate Matern kernel 
    
    Inputs:
    nu: 
    gamma: scaling parameter of Cauchy distribution.
    N: number of random features
    x_train: training samples of shape m x d
    x_test: test samples of shape m' x d
    
    Outputs:
    A_train: Training Random Feature Map A_train
    A_test: Test random feature map A_test
    """
    # number of samples and dimension of features
    m,d = x_train.shape
    rng = np.random.default_rng()
    # random features generated from Cauchy distribution with scaling parameter gamma
    Omega = student(nu,gamma,size=(d,N))
    # Random feature matrix A
    random_offset = rng.uniform(0, 2 * np.pi, size=(1,N))
    A_train = np.cos(x_train@Omega + random_offset)
    
    A_test = np.cos(x_test@Omega + random_offset)
    
    return Omega, random_offset, A_train * (2.0 / N) ** 0.5, A_test * (2.0 / N) ** 0.5

