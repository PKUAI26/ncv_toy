#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:05:37 2018

@author: wanruosi
"""

import tensorflow as tf
import numpy as np

def AC_linear(X_train, F_train, Z_train, X_test, F_test, Z_test, options):
    d, p = Z_train.shape
    
    if options['centering']:
        P = np.eye(d) - np.ones((d, d))/d
        Z_ed = np.matmul(P, Z_train)
        F_ed = np.matmul(P, F_train)
        Sigma_zz = np.matmul(np.transpose(Z_ed), Z_ed)/d
        sigma_zf = np.matmul(F_ed,Z_ed)/d
        a = -np.matmul(np.linalg.inv(Sigma_zz), sigma_zf)
    
    else:
        Sigma_zz = np.matmul(np.transpose(Z_train), Z_train)/d
        sigma_zf = np.matmul(F_train,Z_train)/d
        a = -np.matmul(np.linalg.inv(Sigma_zz), sigma_zf)
    
    ratio1 = np.var(F_train + np.matmul(Z_train, a))/np.var(F_train)
    ratio2 = np.var(F_test + np.matmul(Z_test, a))/np.var(F_test)
    CV_mean = np.mean(np.matmul(Z_train, a))
    CV_test_mean = np.mean(np.matmul(Z_test, a))
    print('linear: train mean {:.6f}, raito {:.6f}; test mean {:.6f}, ratio {:.6f}'.format(CV_mean, ratio1, CV_test_mean, ratio2))
    
    return(np.matmul(Z_train, a), np.matmul(Z_test, a), ratio1, ratio2)

def AC_poly(X_train, F_train, Z_train, X_test, F_test, Z_test, options):
    
    N, p = Z_train.shape
    beta_temp = np.zeros((N, int(p*(p+3)/2)))
    for n in range(N):
        index = 0
        for i in range(p):
            for j in range(i+1):
                if i == j:
                    beta_temp[n, index] = -1/2 + Z_train[n,i] * X_train[n,j]
                else:
                    beta_temp[n, index] = Z_train[n,i] * X_train[n,j] + Z_train[n,j] * X_train[n,i]
                index = index + 1

    beta_temp[n, int(p*(p+1)/2) : int(p*(p+3)/2)] = Z_train[n,:]

    if options['centering']:
        P = np.eye(N) - np.ones((N,N))/N
        Beta_ed = np.matmul(P, beta_temp)
        F_ed = np.matmul(P, F_train)
        Sigma_zz = np.matmul(np.transpose(Beta_ed), Beta_ed)
        Sigma_zf = np.matmul(F_ed, Beta_ed)
        Para = -np.matmul(np.linalg.inv(Sigma_zz), Sigma_zf)
    
    else:
        Sigma_zz = np.matmul(np.transpose(beta_temp), beta_temp)
        Sigma_zf = np.matmul(F_train,beta_temp)
        Para = -np.matmul(np.linalg.inv(Sigma_zz), Sigma_zf)
    
    index = 0
    B = np.zeros((p, p))
    a = np.zeros(p)
    for i in range(p):
        for j in range(i+1):
            B[i,j] = Para[index]
            B[j,i] = Para[index]
            index = index+1
    a = Para[int(p*(p+1)/2):int(p*(p+3)/2)]
    
    CV_train = np.zeros(N)
    for n in range(N):
        CV_train[n] = -1/2*np.trace(B) + np.matmul(a + np.matmul(B, X_train[n,:]), Z_train[n,:])
    
    M = X_test.shape[0]
    CV_test = np.zeros(M)
    for m in range(M):
        CV_test[m] = -1/2*np.trace(B) + np.matmul(a + np.matmul(B, X_test[m,:]), Z_test[m,:])
    #print(CV_test[m])
    
    ratio1 = np.var(F_train + CV_train)/np.var(F_train)
    ratio2 = np.var(F_test + CV_test)/np.var(F_test)
    CV_mean = np.mean(CV_train)
    CV_test_mean = np.mean(CV_test)
    print('Poly: train mean {:.6f}, raito {:.6f}; test mean {:.6f}, ratio {:.6f}'.format(CV_mean, ratio1, CV_test_mean, ratio2))
    
    return(CV_train, CV_test, ratio1, ratio2)

def CF(X0, f0, X1, f1, kernel, opts):
    m = X0.shape[0]
    n = X1.shape[0]
    K0 = np.zeros([m, m])
    K1 = np.zeros([n, m])
    for i in range(m):
        for j in range(m):
            K0[i,j] = kernel(X0[i,:], X0[j,:])
    
    for i in range(n):
        for j in range(m):
            K1[i,j] = kernel(X1[i,:], X0[j,:])


    Eta = np.linalg.inv(K0 + opts['lambda'] * m * np.eye(m))
    cn = np.linalg.norm(K0 + opts['lambda'] * m * np.eye(m), 2) * np.linalg.norm(Eta, 2)

    alpha = np.matmul(np.ones(m), np.matmul(Eta, f0)) / \
    (1+np.matmul(np.ones(m), np.matmul(Eta, np.ones(m))))
    
    f_hat = np.matmul(K1, np.matmul(Eta, f0)) + \
    (np.ones(n) - np.matmul(K1, np.matmul(Eta, np.ones(m)))) * alpha
    
    cv = - f_hat + alpha
    
    ratio = np.var(f1 +cv)/np.var(f1)
    
    print('Control functional {:.6f}, cv mean{:.6f}, ratio {:.6f}'.format(cn, np.mean(cv), ratio))
    
    return cv, ratio

    
class Neural_Stein_CV:
    def __init__(self, reg_para_placeholder, fval_placeholder, 
                 theta_placeholder,  score_placeholder,
                 Optimizer='Adam', layers = [40, 40]):
        
        self.p = theta_placeholder.shape.as_list()[1]
        self.reg_para = reg_para_placeholder
        self.Optimizer = Optimizer
        self.layers = layers        
        
        self.fval = fval_placeholder
        self.theta = theta_placeholder
        self.score = score_placeholder


    def model(self, lr_start):
        phi = self.theta
        score_ = self.score
        
        with tf.variable_scope('NCV'):
            for i in range(len(self.layers)):
                q = phi.shape.as_list()[1]
                with tf.variable_scope('layer-'+str(i+1)):
                    if i < 1:
                        std=1.
                    else:
                        std = 2.
                    W = tf.Variable(tf.random_normal([q, self.layers[i]], 0, np.sqrt(std/self.layers[i])))
                    b = tf.Variable(tf.zeros([self.layers[i]]))
                    phi = tf.sigmoid(tf.matmul(phi, W) + b)
                    
            with tf.variable_scope('output'):
                W = tf.Variable(tf.zeros([self.layers[i], self.p]))
                b = tf.Variable(tf.zeros([self.p]))
                phi = tf.matmul(phi, W) + b
                               
                for i in range(self.p):
                    tf.add_to_collection('g2', tf.gradients(phi[:,i], self.theta)[0][:,i])
        
                Stein = tf.add_n(tf.get_collection('g2'))
                self.cv = Stein + tf.reduce_sum(tf.multiply(phi, score_), 1)
                

            self.mu = tf.Variable(0, dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(self.fval+self.cv+self.mu))
            self.reg = self.reg_para * tf.reduce_mean(tf.square(self.cv))
            self.cost = self.loss + self.reg
    
        learning_rate = tf.train.exponential_decay(lr_start, 20000, 2000, 0.8, staircase=True)
        if self.Optimizer == 'Adam':
            self.train_step = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon=0.0001).minimize(self.cost)
        
        elif self.Optimizer == 'SGD':
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.cost)

