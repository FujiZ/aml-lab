# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
from threading import Thread
import functools
import time

# (global) variable definition here
TRAINING_TIME_LIMIT = 60 * 10
A = None  # global matrix for training
LEARNING_RATE = 0.005
EPOCH = 750
BATCH = 500


# class definition here

# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    def squared_norm(mat, axis=None):
        return np.sum(mat ** 2, axis)

    data = traindata[0]  # shape in (num, dim)
    label = traindata[1]
    num, dim = data.shape
    global A, BATCH
    A = np.identity(dim, dtype=np.float)
    batch = min((BATCH, num))
    print('batch:', batch, 'epoch:', EPOCH, 'learning rate:', LEARNING_RATE)
    print('start time:', time.ctime())
    for i in range(EPOCH):
        # select BATCH sample randomly from traindata
        idx = np.random.choice(num, batch, replace=False)
        x = data[idx, :].T
        y = label[idx]
        mask = (y[:, np.newaxis] == y)
        a_x = np.dot(A, x)  # a_x = A·(x1, ..., xn)
        # exp_dist = exp(-||Ax_i-Ax_j||^2)  n*n matrix
        exp_dist_ij = np.exp(-squared_norm(a_x[:, :, np.newaxis] - a_x[:, np.newaxis, :], axis=0))
        np.fill_diagonal(exp_dist_ij, 0)  # set exp_dist_ii to 0
        sum_exp_dist_ij = np.sum(exp_dist_ij, axis=0)[:, np.newaxis]
        sum_exp_dist_ij[sum_exp_dist_ij == 0.0] = 1
        p_ij = exp_dist_ij / sum_exp_dist_ij
        p_i = np.sum(p_ij * mask, axis=1)  # ATTENTION: axis=1 because p_ij isn't symmetrical!
        # x_i - x_j in n*n*d matrix
        x_ij = np.stack(x[:, :, np.newaxis] - x[:, np.newaxis, :], axis=2)
        # n*n*d*d tensor for x_ij*x_ij^T
        x_xt = np.matmul(x_ij[..., np.newaxis], x_ij[..., np.newaxis, :])
        # fa = np.sum(p_i)
        p_x_xt = (p_ij[:, :, np.newaxis, np.newaxis] * x_xt)  # p_ik*x_ik*x_ik^T
        # p_i*sum(p_ik*x_ik*x_ik^T)
        p_sum_p_x_xt = p_i[:, np.newaxis, np.newaxis] * p_x_xt.sum(axis=1)
        # sum(p_ij*x_ij*x_ij^T) (j in C_i)
        sum_p_x_xt = (mask[:, :, np.newaxis, np.newaxis] * p_x_xt).sum(axis=1)
        df_da = 2 * np.dot(A, (p_sum_p_x_xt - sum_p_x_xt).sum(axis=0))
        A += LEARNING_RATE * df_da
    print('stop time:', time.ctime())
    return 0


def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)


def distance(inst_a, inst_b):
    diff_xy = (inst_a - inst_b)[:, np.newaxis]
    a_xy = np.dot(A, diff_xy)
    return np.sqrt(np.dot(a_xy.T, a_xy))


# main program here
if __name__ == '__main__':
    pass
