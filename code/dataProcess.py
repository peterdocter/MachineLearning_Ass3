#coding=utf-8
'''
Created on 2016��6��9��

@author: pangb
'''
def loadData1(filename):
    fr = open(filename)
    numFeat = len(fr.readline().strip().split(','))-3
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curLine[i+2]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def loadData2(filename):
    fr = open(filename)
    numFeat = len(fr.readline().strip().split(','))-1
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curline = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curline[i+1]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[0]))
    return dataMat, labelMat
            
            