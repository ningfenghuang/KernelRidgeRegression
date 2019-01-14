#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:17:36 2018

@author: Qifan Huang
"""


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


class KernelRidgeRegression():
    def __init__(self, numSample):
        """
        Input: numSample: number of sample size 
        """
        self.__numSample = numSample
        self.__gammaPath = [10**(i+2) for i in range(5)]
        self.__lambOnePath = [10**(-i) for i in range(5)]
        self.__lambTwoPath = [10**(-i) for i in range(5)]
        
    @property
    def getParamTrue(self):
        """
        Use: accessor of class RegressionSideInfo's field 
        """
        return  self.__numSample
    
    
    def trueFunction(self, x):
        """
        Returns: an int, value of true f(x)
        """
        thresholdList = np.array([0.2, 0.4, 0.6, 0.8])
        fTrue = 10 * np.sum(x >= thresholdList)
        return fTrue
    
    
    def getData(self):
        """
        data generation process with one outlier 
        Returns: 4 arrays 
        """
        ## 创建等差数列
        self.xSample = np.linspace(0, 1, self.__numSample)[:, np.newaxis]
        self.fSample = np.array([ self.trueFunction(x) for x in self.xSample])[:, np.newaxis]
        self.eSample = np.random.randn(self.__numSample)[:, np.newaxis]
        self.ySample = self.fSample + self.eSample
        ## outlier 
        self.ySample[24] = 0
        return self.xSample, self.fSample, self.eSample, self.ySample
    
    
    def kernelFunction(self, param, xTest, xTrain):
        """
        Use: calculate rbf kernel
        Input: (1) param
               (2) test feature array with dimension numTest * 1 (only 1 feature)
               (3) train feature array with dimension numTrain * 1 
        Return: kernel array with dimension numTest * numTrain 
        """
        kernel = np.exp(-param * (xTest - xTrain.T)**2)
        return kernel 
        
    
    def objectiveFunction(self, kernel, y, alpha, lambOne, lambTwo, typeObj):
        """
        Input:(1) kernel: a numTrain * numTrain array
              (2) y
              (3) alpha: a numTrain * 1 array 
              (4) lambOne, lambTwo: two hyper parameters 
              (5) typeObj = 0: least squares loss + L2 regularizer
                  typeObj = 1: huber loss + L2 regularizer
                  typeObj = 2: least squares loss + TV regularizer + L2 regularizer
                  typeObj = 3: least squares loss + L2 regularizer + non-negative constraints 
        Return: a cp.Minimize class, which defines sample objective function 
                a list, which defines parameter constraints 
        """
        constraints = []
        if typeObj == 0:
            objective = cp.Minimize(cp.sum_squares(kernel*alpha - y) 
                                    + lambOne * cp.quad_form(alpha, kernel))
        if typeObj == 1:
            objective = cp.Minimize(cp.sum_entries(cp.huber(kernel*alpha - y, 1))
                                    + lambOne * cp.quad_form(alpha, kernel))
        if typeObj == 2:
            objective = cp.Minimize(cp.sum_squares(kernel*alpha - y) 
                                     + lambOne * cp.norm(((kernel*alpha)[1::]-(kernel*alpha)[:-1:]),1)
                                     + lambTwo * cp.quad_form(alpha, kernel))
        if typeObj == 3:
            objective = cp.Minimize(cp.sum_squares(kernel*alpha - y) 
                                    + lambOne * cp.quad_form(alpha, kernel))
            constraints = [ (kernel*alpha)[1::] - (kernel*alpha)[:-1:] >= 0 ]
        return objective, constraints
    
    
    
    
    def cvLooLoss(self, kernel, y, lambOne, lambTwo, typeObj):
        """
        Return: an int, cvLooLoss given lambOne, lambTwo and gamma (shown in kernel)
        """
        numSample = kernel.shape[0]
        alpha = cp.Variable(numSample - 1)
        loss = 0
    
        for j in range(numSample):
            ## delete the jth observations
            kernelTrain = np.delete(kernel, j, 0)
            kernelTrain = np.delete(kernelTrain, j, 1)
            yTrain = np.delete(y, j)
            objective, constraints = self.objectiveFunction(kernelTrain, yTrain, alpha, 
                                                       lambOne, lambTwo, typeObj)
            ## create a Problem class
            if typeObj == 3:
                problem = cp.Problem(objective, constraints)
            else:
                problem = cp.Problem(objective)
            problem.solve()
            
            kernelTest = np.delete(kernel[j], j)
            fhat = kernelTest @ alpha.value
            #print(fhat)
            loss += (y[j] - fhat)**2
        return loss    
    
        
    
    def findHyperParam(self, xTrain, y, typeObj):
        """
        Return: three doubles: optimal gamma, optimal lambOne and optimal lambTwo
        """
        lossArray = np.zeros((len(self.__gammaPath), len(self.__lambOnePath), 
                              len(self.__lambTwoPath)))
        
        for i, gamma in enumerate(self.__gammaPath):
            kernel = self.kernelFunction(gamma, xTrain, xTrain)
            for j, lambOne in enumerate(self.__lambOnePath):
                if typeObj == 2:
                    for r, lambTwo in enumerate(self.__lambTwoPath):
                        lossArray[i][j][r] = self.cvLooLoss(kernel, y, 
                                                       lambOne, lambTwo, typeObj)
                else:
                    lossArray[i][j] = self.cvLooLoss(kernel, y, lambOne, 1, typeObj)
                    
        gammaOptIndex = np.where(lossArray == np.min(lossArray))[0][0]
        lambOneOptIndex = np.where(lossArray == np.min(lossArray))[1][0]
        lambTwoOptIndex = np.where(lossArray == np.min(lossArray))[2][0]
        self.gammaOpt = self.__gammaPath[gammaOptIndex]
        self.lambOneOpt = self.__lambOnePath[lambOneOptIndex]
        self.lambTwoOpt = self.__lambTwoPath[lambTwoOptIndex]
        print([self.gammaOpt, self.lambOneOpt, self.lambTwoOpt])
        return self.gammaOpt, self.lambOneOpt, self.lambTwoOpt
    
    
    
    def getPrediction(self, x, y, typeObj):
        """
        Returns: 
        """
        kernel = self.kernelFunction(self.gammaOpt, x, x)
        numSample = kernel.shape[0]
        alpha = cp.Variable(numSample )
        objective,  constraints = self.objectiveFunction(kernel, y, alpha, self.lambOneOpt, 
                                           self.lambTwoOpt, typeObj)
        if typeObj == 3:
            problem = cp.Problem(objective, constraints)
        else:
            problem = cp.Problem(objective)
        problem.solve()
        fHat = kernel @ alpha.value
        
        #print(fHat)
        return fHat
    
    
    def getPlot(self, x, y, fTrue, fHat):
        """
        """
        plt.scatter(x, y, label ="data" )
        plt.plot(x, fTrue, label = "f_true")
        plt.plot(x, fHat, label = "f_hat")
        plt.legend()
        plt.show()
        
        
        
        
 
