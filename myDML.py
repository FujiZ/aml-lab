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
STEP = 0.01
TIME = 2000
BATCH = 200


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
    def squared_norm(mat, axis=0):
        return np.sum(mat ** 2, axis)

    x = traindata[0].T  # shape in (dim, sample)
    label = traindata[1]
    dim, num = x.shape
    global A
    A = np.identity(dim, dtype=np.float)  # TODO change dtype to float32?
    mask = (label[:, np.newaxis] == label)
    # x_i - x_j in n*n*d matrix
    x_ij = np.stack(x[:, :, np.newaxis] - x[:, np.newaxis, :], axis=2)
    # n*n*d*d tensor for x_ij*x_ij^T
    x_xt = np.matmul(x_ij[..., np.newaxis], x_ij[..., np.newaxis, :])
    # TODO iter the code below
    for i in range(TIME):
        a_x = np.dot(A, x)  # a_x = A·(x1, ..., xn)
        # exp_dist = exp(-||Ax_i-Ax_j||^2)
        exp_dist_ij = np.exp(-squared_norm(a_x[:, :, np.newaxis] - a_x[:, np.newaxis, :], axis=0))
        p_ij = exp_dist_ij / np.sum(exp_dist_ij, axis=0)
        p_i = np.sum(p_ij * mask, axis=0)
        # fa = np.sum(p_i)
        p_x_xt = (p_ij[:, :, np.newaxis, np.newaxis] * x_xt)  # p_ik*x_ik*x_ik^T
        # p_i*sum(p_ik*x_ik*x_ik^T)
        p_sum_p_x_xt = p_i[:, np.newaxis, np.newaxis] * p_x_xt.sum(axis=1)
        # sum(p_ij*x_ij*x_ij^T) (j∈C_i)
        sum_p_x_xt = (mask[:, :, np.newaxis, np.newaxis] * p_x_xt).sum(axis=1)
        # TODO calc df_da
        df_da = 2 * A * (p_sum_p_x_xt - sum_p_x_xt).sum(axis=0)
        A += STEP * df_da
        print(A)
        print(time.ctime())
    return 0


def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)


def distance(inst_a, inst_b):
    diff_ab = np.dot(A, inst_a[:, np.newaxis] - inst_b[:, np.newaxis])
    return np.sqrt(np.dot(diff_ab.T, diff_ab))


# main program here
if __name__ == '__main__':
    pass
