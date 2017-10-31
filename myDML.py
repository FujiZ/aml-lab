# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
from threading import Thread
import functools

# (global) variable definition here
TRAINING_TIME_LIMIT = 60 * 10
A = None  # global matrix for training


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
    # 在此处完成你的训练函数，注意训练时间不要超过TRAINING_TIME_LIMIT(秒)。
    # np.dot(mata, matb)
    # np.exp()
    # narray.sum()
    # matb.T return (matb)T
    # np.identity(n) init
    # A = np.identity(traindata[0])
    dataset = traindata[0]
    labelset = traindata[1]
    global A
    dim = dataset.shape[1]  # dim of each entry
    A = np.identity(dim, dtype=np.float64)  # TODO change dtype to float32?
    print(traindata[1])
    return 0


def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)


def distance(inst_a, inst_b):
    dist = np.random.rand()  # 随机度量 这是一个错误示范，请删除这行 _(:3 」∠)_
    return dist


# main program here
if __name__ == '__main__':
    pass
