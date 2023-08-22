# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:07:45 2023

@author: visha
"""
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import sys
import pickle
import os
import ast
from pandas import read_csv
import gensim
from numpy.linalg import norm
import setting
TESTDATASET=["senseval2","senseval3","semeval2007","semeval2013","semeval2015"]
algoListLabel=["Algo 1: wm","Algo 2: wm (cloud context)","Algo 3: Word2Vec","Algo 4: MFS"]
TRAINSET=["semcor"]
MFS_TEST_METRIC=[0.6291,0.61655,0.55470,0.625,0.599]
MFS_TRAIN_METRIC=0.6989
import setting
TRAIN=1
if TRAIN==1:
    sizes=[7,9,11,13,15,17]
    algoList=["predict_wm_train_data","predict_wm_recursion_train_data","predict_word2vec_train_data","MFS"] 
    semcor={}
    for algoName in algoList:  
        semcor[algoName]={}
else:
    sizes=[5,7,9,11,13,15,17]
    algoList=["predict_wm_test_data","predict_wm_recursion_test_data","predict_word2vec_test_data","MFS"]
    testDict={}
    senseval2={}
    senseval3={}
    semeval2007={}
    semeval2013={}
    semeval2015={}
    for testSet in TESTDATASET:
        testDict[testSet]={}
        for algoName in algoList:  
            testDict[testSet][algoName]={}
       
for algoName in algoList:        
    #for testDataset in setting.TESTDATASET:
        #testData="_{}_window_{}_metric.csv".format(os.path.basename(testDataset).split(".")[0].split("_")[0],setting.WINDOWSIZE)
    for size in sizes:
        metricPath="./window_{}/metric/".format(size)
        #fileName=metricPath+algoName+"_{}_window_{}_metric.csv".format(os.path.basename(testDataset).split(".")[0].split("_")[0],size)
        fileName=metricPath+algoName+"_window_{}_metric.csv".format(size)
        print(fileName)
        if os.path.isfile(fileName) or algoName == "MFS":
            if os.path.isfile(fileName):
                df1 = read_csv(fileName, sep="\t", encoding="utf-8",index_col=0)
            if TRAIN==1:
                if algoName!="MFS":
                    semcor[algoName][size]=df1["train"]["wm_micro"]
                else:
                    semcor[algoName][size]=MFS_TRAIN_METRIC
            else:
                for i,testSet in enumerate(TESTDATASET):
                    if algoName!="MFS":
                        testDict[testSet][algoName][size]=df1[testSet]["wm_micro"]
                    else:
                        testDict[testSet][algoName][size]=MFS_TEST_METRIC[i]
                

if TRAIN==1:
    from matplotlib import pyplot as plt
    for i,algoName in enumerate(algoList):  
        x = semcor[algoName].keys()
        y = semcor[algoName].values()
        plt.plot(x,y, label=algoListLabel[i])
    plt.title("semcor")
    plt.ylabel('Accuracy')
    plt.xlabel('window size')
    plt.legend()
    plt.savefig("train.png")
    plt.show()
    
else:
    once=0
    from matplotlib import pyplot as plt
    figure, axis = plt.subplots(1, 5,figsize=(20,5),sharex=True,sharey=True)
    for j,testSet in enumerate(TESTDATASET):
        for i,algoName in enumerate(algoList):  
            x = testDict[testSet][algoName].keys()
            y = testDict[testSet][algoName].values()
            axis[j].plot(x,y,label=algoListLabel[i])
        #
        axis[j].set_title(testSet)
        figure.text(0.5, 0.0, 'window size', ha='center')
        figure.text(0.0, 0.5, 'Accuracy', va='center', rotation='vertical')
        #axis[j].set_ylabel('Accuracy')
        #axis[j].set_xlabel('window size')
        #axis[j].legend()
    figure.legend(algoListLabel,loc='upper left')
    figure.tight_layout()
    plt.savefig("test.png")
    plt.show()