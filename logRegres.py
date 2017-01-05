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
    :return: 数据集列表，标签集列表
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


def plotBestFit(weights):
    '''
    画出数据集和Logistic回归最佳拟合直线函数
    :param weights:
    :return:
    '''
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] # 数据集数组行数

    # 数据集按类别分类
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 数据值为1的用红色方块标记
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    # 数据值为0的用绿色圆点标记
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 生成从-3到+3步长为0.1的数组
    x = np.arange(-3.0, 3.0, 0.1)
    # 按照阶跃函数等于0.5时划分分类，此时阶跃函数自变量即回归直线为0，即w0x0+w1x1+w2x2=0，解x2为：
    y = (-weights[0]-weights[1]*x)/weights[2]
    # 纵标y即为特征值x2（x0==1）
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 标签集
    :return: 权重系数向量
    '''
    # 全为数组间运算（对应相乘）
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n) # n列的行向量（数组）
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) # 数组间元素对应相乘，获得结果为数值
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def step01():
    '''
    获取权重系数列向量
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    print(weights)

def step02():
    '''
    画出决策边界
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = gradAscent(dataArr, labelMat)
    # weights.getA() 矩阵转数组
    plotBestFit(weights.getA())


def step03():
    '''
    使用随机梯度上升算法优化
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent0(np.array(dataArr), labelMat)
    plotBestFit(weights)

if __name__ == '__main__':
    # step01()
    # step02()
    step03()