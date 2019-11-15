#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:54:33 2018

@author: ruos
"""

import numpy as np
import tensorflow as tf
import utils
import CVfunctions as cvf
import os
import argparse


def kernel(x0, x1, alpha1=0.1, alpha2=1):
    return 1/(1+alpha1*np.dot(x0, x0)+alpha1*np.dot(x1, x1)) * np.exp(-np.dot(x1-x0,x1-x0)/(2*alpha2**2))

def expand(data, e):
    
    return np.concatenate([data]*e, axis=0)

def summary(fval, expand):
    N = int(fval.shape[0]/expand)
    fval_mean = 0
    for i in range(expand):
        fval_mean += fval[i*N:(i+1)*N]
    
    return fval_mean/expand


def main():
    parser = argparse.ArgumentParser(description = 'control variates experiment')
    parser.add_argument('--method', choices=['linear', 'poly', 'cf', 'NSCV'], default='NSCV')
    
    parser.add_argument('--layers', default=[40, 40, 40]) #NSCV
    parser.add_argument('--lr_init', type=float, default = 0.008)
    parser.add_argument('--optimizer', choices=['SGD', 'Mom', 'Adam'], default='Adam')
    parser.add_argument('--reg_para', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=50000)
    
    parser.add_argument('--Dim', type=int, default=10)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--valid_size', type=int, default=500)
    
    parser.add_argument('--pre_mu', action='store_true', default=False, help='Use two training stage to determine the start value of mu')
    parser.add_argument('--mu', default=-5., help='Start value of mu, usually input the mean of samples')
    parser.add_argument('--save_model', default='')
    args = parser.parse_args()
    
    
    d = args.Dim
    M = args.valid_size
    N = args.train_size

    X0 = np.random.normal(-0.5, 1, (N+M, d))
    X1 = np.random.normal(0.5, 1, (N+M, d))
    index = np.repeat(np.random.binomial(1, 0.5, (N+M,1)), d, 1)
    X = X0 * index + X1 *(1-index)
    F = np.sin(np.pi/d*np.sum(X,1)) + 5
    Z = np.apply_along_axis(utils.score, 1, X)
    
    X_train, F_train, Z_train, X_valid, F_valid, Z_valid  = utils.Seperate_data(N, X, F, Z)
    
    
    if args.method == 'linear':
        opts={'centering': False}
        cvf.AC_linear(X_train, F_train, Z_train, X_valid, F_valid, Z_valid, opts)
        
    if args.method == 'poly':
        
        opts={'centering': True}
        cvf.AC_poly(X_train, F_train, Z_train, X_valid, F_valid, Z_valid, opts)
        
    if args.method == 'cf':
        opts = {'lambda': 1e-5}
        cvf.CF(X_train, F_train, X_valid, F_valid, kernel, opts)
    
    if args.method == 'NSCV':
        tf.reset_default_graph()
        fval = tf.placeholder(tf.float32, shape=(None))
        theta = tf.placeholder(tf.float32, shape=(None, d))
        score = tf.placeholder(tf.float32, shape=(None, d))
        reg_para = tf.placeholder(tf.float32)
        
        CV = cvf.Neural_Stein_CV(reg_para_placeholder=reg_para, fval_placeholder = fval,
                                  theta_placeholder = theta, score_placeholder = score,
                                  Optimizer=args.optimizer, layers = args.layers)
        CV.model(args.lr_init)
        

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('iter, mu, loss_t, train ratio, loss_v, test ratio')
            step = 0
            if args.pre_mu:
                phase = 'pre-trained'
            else:
                phase = 'training'
                sess.run(CV.mu.assign(args.mu))
            while step < args.max_iter:
                x_example, f_example, z_example = utils.Get_batch(X_train, F_train, Z_train, args.batch_size)
                if step < 15000 and phase == 'pre-trained':
                    lambda_ = 10
                    CV.train_step.run(feed_dict={theta: x_example, fval: f_example, score: z_example, 
                                                 reg_para: lambda_})
                
                elif phase == 'pre-trained':
                    phase = 'training'
                    mu_ = sess.run(CV.mu)
                    print('==========================================================================')
                    print('pre-trained phase done, initalize the model and load the pre-trained mu {}'.format(mu_))
                    print('==========================================================================')
                    sess.run(tf.global_variables_initializer())
                    sess.run(CV.mu.assign(mu_))
                    step = 0
                    continue
                    
                else:
                    lambda_ = args.reg_para
                    CV.train_step.run(feed_dict={theta: x_example, fval: f_example, score: z_example, 
                                                 reg_para: lambda_})
        
                if (step+1)%500 == 0:
                    cv_t, loss_t = sess.run([CV.cv, CV.cost], feed_dict={theta: X_train, fval: F_train, score: Z_train, reg_para: lambda_})
                    cv_v, loss_v = sess.run([CV.cv, CV.cost], feed_dict={theta: X_valid, fval: F_valid, score: Z_valid, reg_para: lambda_})
                    mu_ = sess.run(CV.mu)
                    ratio = np.var(F_valid+cv_v)/np.var(F_valid)
                    print('{} phase: {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(phase, step, 
                          mu_, loss_t, np.var(F_train+cv_t)/np.var(F_train), loss_v, ratio))
                    
                step += 1
        

        
    if args.save_model is not '':
        saver = tf.train.Saver()
        saver.save(sess, args.save_model)
        
        
if __name__ == "__main__":
    main()
        
    
    
