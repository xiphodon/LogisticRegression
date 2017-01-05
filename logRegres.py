#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/5 14:27
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : logRegres.py
# @Software: PyCharm


# 逻辑回归算法

import numpy as np

def loadDataSet():
    '''
    加载数据集
    :return: 数据集，标签集
    '''
    dataMat = []; labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    '''
    阶跃函数
    :param inX: 自变量x
    :return: 因变量y
    '''
    return 1.0/(1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法（同梯度下降算法，获得使代价函数最小的权重系数列向量），获取权重系数列向量
    :param dataMatIn: 数据集列表
    :param classLabels: 标签集列表
    :return: 权重系数列向量
    '''
    dataMatrix = np.mat(dataMatIn) # 100*3 数据集矩阵
    labelMat = np.mat(classLabels).transpose() # 100*1 标签列向量
    m,n = np.shape(dataMatrix)
    alpha = 0.001 # 学习率(步长)
    maxCycles = 500 # 最大循环次数(迭代次数)
    weights = np.ones((n,1)) # 3*1 权重系数列向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # 矩阵乘法，得到 100*1 的预测值列向量
        error = (labelMat - h) # 矩阵减法，得到 100*1 预测误差列向量
        weights = weights + alpha * dataMatrix.transpose()* error # 矩阵乘法，更新权重系数列向量

        # 梯度下降算法
        #  weights = weights - alpha * dataMatrix.transpose()* error

    return weights


def step01():
    '''
    获取权重系数列向量
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    print(gradAscent(dataArr, labelMat))


if __name__ == '__main__':
    step01()