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
    :return: 权重系数列向量（回归系数）
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
    :return: 权重系数向量（回归系数）
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


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    改进的随机梯度上升算法
    :param dataMatrix: 数据集
    :param classLabels: 标签集
    :param numIter: 迭代次数
    :return: 回归系数
    '''
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 # 学习率渐渐变小，趋于收敛，避免严格下降，使代价函数收敛更快，避免高频波动
            randIndex = int(np.random.uniform(0,len(dataIndex))) # 随机选取数据，减小代价函数周期性波动
            h = sigmoid(np.sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #选择完剔除该数据索引
    return weights


def classifyVector(inX, weights):
    '''
    分类器
    :param inX: 测试数据（一条）
    :param weights: 回归系数
    :return:
    '''
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    '''
    从疝气病症预测病马的死亡率——分类错误率
    通过训练集训练后分类测试集的错误率
    :return: 错误率
    '''
    frTrain = open('data/horseColicTraining.txt');
    frTest = open('data/horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 训练的逻辑回归系数
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            # 若分类器分类错误，错误数+1
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    '''
    调用colicTest()方法多次后求错误率平均值
    :return: 平均错误率
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


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

def step04():
    '''
    使用改进的随机梯度上升算法
    :return:
    '''
    dataArr, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataArr), labelMat, 200)
    plotBestFit(weights)


if __name__ == '__main__':
    # step01()
    # step02()
    # step03()
    # step04()
    colicTest()
    multiTest()