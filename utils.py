#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:37:45 2017

@author: ruos
"""

import numpy as np

def score(x):
    a0 = np.exp(-np.sum(np.square(x+0.5))/2)
    a1 = np.exp(-np.sum(np.square(x-0.5))/2)
    return -(a0*(x+0.5) + a1*(x-0.5))/(a0+a1)

def Seperate_data(d, X_all, F_all, Z_all):
    l = X_all.shape
    index = np.arange(0,l[0])
    np.random.shuffle(index)
    
    X_temp = X_all[index,:]
    F_temp = F_all[index]
    Z_temp = Z_all[index,:]
    
    X_train = X_temp[:d,:]
    F_train = F_temp[:d]
    Z_train = Z_temp[:d,:]
    
    X_test = X_temp[d:,:]
    F_test = F_temp[d:]
    Z_test = Z_temp[d:,:]
    return X_train, F_train, Z_train, X_test, F_test, Z_test 

def Get_batch(X, F, Z, batch_size):
    l = X.shape
    index = np.random.choice(l[0], size = batch_size, replace = False)
    
    X_batch = X[index,:]
    Z_batch = Z[index,:]
    F_batch = F[index]
    
    return X_batch, F_batch, Z_batch


