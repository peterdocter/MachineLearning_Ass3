#coding=utf-8

'''
Created on 2016��6��9��

@author: pangb
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation
import numpy
import dataProcess
import math

def regularize(xArr):
    xMat = numpy.mat(xArr)
    inMeans = numpy.mean(xMat, 0)
    inVar = numpy.var(xMat, 0)
    xMat = (xMat - inMeans) / inVar
    
    return xMat.tolist()
'''
def diff1(y1, y2):
    lenth = len(y1)
    count = 0
    for i in range(lenth):
        if y1[i] > 0.5 and y2[i] > 0.5:
            count += 1
        if y1[i] < 0.5 and y2[i] < 0.5:
            count += 1
    return count
'''

def diff1(y1, y2):
    length = len(y1)
    count = 0
    for i in range(length):
        if y1[i] == y2[i]:
            count += 1
    return count 

class svmLearn:
    def __init__(self, x, y):
        # x和y的矩阵
        self.xArr = x
        self.yArr = y
        # 根据集合分开训练集和测试集
        rowNum = len(y)
        singleNum = math.ceil(rowNum / 5)
        testNum = singleNum * 4
        self.TestXarr = []
        self.TestYarr = []
        self.TrainXarr = []
        self.TrainYarr = []
        trainNum = rowNum - testNum;
        for i in range(int(singleNum)):
            if 5*i >= rowNum -1:
                break
            self.TrainXarr.append(x[5*i])
            self.TrainYarr.append(y[5*i])
            if 5*i+1 >= rowNum -1:
                break
            self.TrainXarr.append(x[5*i+1])
            self.TrainYarr.append(y[5*i+1])
            if 5*i+2 >= rowNum -1:
                break
            self.TrainXarr.append(x[5*i+2])
            self.TrainYarr.append(y[5*i+2])
            if 5*i+3 >= rowNum -1:
                break
            self.TrainXarr.append(x[5*i+3])
            self.TrainYarr.append(y[5*i+3])
            if 5*i+4 >= rowNum -1:
                break
            self.TestXarr.append(x[5*i+4])
            self.TestYarr.append(y[5*i+4])
        
        '''self.TestXarr = x[0:testNum][:]
        self.TrainXarr = x[testNum: rowNum][:]
        self.TestYarr = y[0:testNum]
        self.TrainYarr = y[testNum:rowNum]'''
    
    def linearSVM(self):
        #print numpy.shape(self.TrainXarr)
        bestResult = 0
        besti = 0
        bestj = 0
        plotx = [0] * 64
        ploty = [0] * 64
        plotz = [0] * 64
        index = 0
        for i in range(8):
            for j in range(8):
                
                
                linear_svc = svm.SVC(kernel = 'linear', C= 10 ** (i - 5), gamma = 10** (j - 5))
        #linear_svc.fit(self.TrainXarr, self.TrainYarr)
        #yPredict = linear_svc.predict(self.TestXarr)
        #print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        #print len(yPredict)
                scores = cross_validation.cross_val_score(linear_svc, self.TrainXarr, self.TrainYarr, cv = 5)
                #print i, j
                #print scores
                plotx[index] = i - 5
                ploty[index] = j - 5
                plotz[index] = scores.mean()
                index += 1
                if scores.mean() > bestResult:
                    bestResult = scores.mean()
                    besti = i
                    bestj = j
        
        print bestResult
        print besti
        print bestj
        linear_svc = svm.SVC(kernel = 'linear', C= 10 ** (besti - 5), gamma = 10** (bestj - 5))
        linear_svc.fit(self.TrainXarr, self.TrainYarr)
        yPredict = linear_svc.predict(self.TestXarr)
        print '测试集正确率:'
        print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.plot_trisurf(plotx, ploty, plotz)
        ax.plot_trisurf(plotx, ploty, plotz,cmap=cm.jet, linewidth=0.2)
        plt.xlabel('C:')
        plt.ylabel('gamma:')
        plt.show()
        
    def rbfSVM(self):
        bestResult = 0
        besti = 0
        bestj = 0
        plotx = [0] * 64
        ploty = [0] * 64
        plotz = [0] * 64
        index = 0
        for i in range(8):
            for j in range(8):
                
                linear_svc = svm.SVC(kernel = 'rbf', C= 10 ** (i - 5), gamma = 10** (j - 5))
        #linear_svc.fit(self.TrainXarr, self.TrainYarr)
        #yPredict = linear_svc.predict(self.TestXarr)
        #print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        #print len(yPredict)
                scores = cross_validation.cross_val_score(linear_svc, self.TrainXarr, self.TrainYarr, cv = 5)
                #print i,j
                #print scores
                plotx[index] = i - 5
                ploty[index] = j - 5
                plotz[index] = scores.mean()
                index += 1
                if scores.mean() > bestResult:
                    bestResult = scores.mean()
                    besti = i
                    bestj = j
        
        print bestResult
        print besti
        print bestj
        linear_svc = svm.SVC(kernel = 'rbf', C= 10 ** (besti - 5), gamma = 10** (bestj - 5))
        linear_svc.fit(self.TrainXarr, self.TrainYarr)
        yPredict = linear_svc.predict(self.TestXarr)
        print '测试集正确率:'
        print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.plot_trisurf(plotx, ploty, plotz)
        ax.plot_trisurf(plotx, ploty, plotz,cmap=cm.jet, linewidth=0.2)
        #ax.plot_surface(plotx, ploty, plotz, rstride=1, cstride=1, cmap='rainbow')
        plt.xlabel('C:')
        plt.ylabel('gamma:')
        plt.show()
        
    def sigmoidsvm(self):
        bestResult = 0
        besti = 0
        bestj = 0
        plotx = [0] * 64
        ploty = [0] * 64
        plotz = [0] * 64
        index = 0
        for i in range(8):
            for j in range(8):
                
                linear_svc = svm.SVC(kernel = 'sigmoid', C= 10 ** (i - 5), gamma = 10** (j - 5))
        #linear_svc.fit(self.TrainXarr, self.TrainYarr)
        #yPredict = linear_svc.predict(self.TestXarr)
        #print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        #print len(yPredict)
                scores = cross_validation.cross_val_score(linear_svc, self.TrainXarr, self.TrainYarr, cv = 5)
                plotx[index] = i - 5
                ploty[index] = j - 5
                plotz[index] = scores.mean()
                index += 1
                if scores.mean() > bestResult:
                    bestResult = scores.mean()
                    besti = i
                    bestj = j
        
        print bestResult
        print besti
        print bestj
        linear_svc = svm.SVC(kernel = 'sigmoid', C= 10 ** (besti - 5), gamma = 10** (bestj - 5))
        linear_svc.fit(self.TrainXarr, self.TrainYarr)
        yPredict = linear_svc.predict(self.TestXarr)
        print '测试集正确率:'
        print diff1(yPredict, self.TestYarr) / float(len(yPredict))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.plot_trisurf(plotx, ploty, plotz)
        ax.plot_trisurf(plotx, ploty, plotz,cmap=cm.jet, linewidth=0.2)
        #ax.plot_surface(plotx, ploty, plotz, rstride=1, cstride=1, cmap='rainbow')
        plt.xlabel('C:')
        plt.ylabel('gamma:')
        plt.show()

if __name__ == '__main__':
    
    xArr, yArr = dataProcess.loadData2('wine.data.txt')
    #xArr, yArr = dataProcess.loadData1('clean1.data')
    #xArr = preprocessing.scale(xArr)
    svmle = svmLearn(xArr, yArr)
    print 'Linearsvm:'
    svmle.linearSVM()
    print 'RBFSVM:' 
    svmle.rbfSVM()
    print 'sigmoid'
    svmle.sigmoidsvm()