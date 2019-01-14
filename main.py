#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:17:36 2018

@author: Qifan Huang
"""


import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from KernelRidgeRegression import KernelRidgeRegression



def main():
    typeObj = 0
    example = KernelRidgeRegression(50)
    xSample, fSample, eSample, ySample = example.getData()
    
    
    example.findHyperParam(xSample, ySample, 0)
    fHat = example.getPrediction(xSample, ySample, 0)
    print(np.mean(fHat))
    example.getPlot(xSample, ySample, fSample, fHat)
    
    example.findHyperParam(xSample, ySample, 1)
    example.lambTwoOpt = 0.1
    failed = True
    while failed:
        try:
            fHat = example.getPrediction(xSample, ySample, 1)
            failed = False
        except:
            pass
    print(np.mean(fHat))
    example.getPlot(xSample, ySample, fSample, fHat)
    
    
    
    example.findHyperParam(xSample, ySample, 2)
    fHat = example.getPrediction(xSample, ySample, 2)
    print(np.mean(fHat))
    example.getPlot(xSample, ySample, fSample, fHat)
   

    
    example.findHyperParam(xSample, ySample, 3)
    fHat = example.getPrediction(xSample, ySample, 3)
    print(np.mean(fHat))
    example.getPlot(xSample, ySample, fSample, fHat)
    

    
if __name__ == '__main__':
   main()    