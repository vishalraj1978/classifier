# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:31:05 2023

@author: visha
"""
import sys
import pandas as pd
import re

import pickle
#import subprocess
import os
#from pandas import read_csv   
#import ast

#TRAIN_DATASET="semeval2013_instance_compact"

import  create_vocab 
import setting
import numpy as np
from nltk.corpus import stopwords
#stopWordList = list(stopwords.words('english'))
#stopWordSet = set(stopwords.words('english'))



def normaliseWeightMatrix(f,weightMatrix, vocab, track):
    '''This matrix takes the inputs W matrix,V(output of create_V()), and  normalizes the W matrix by dividing the
    co-occurence count value of each cell in the matrix by product of their corresponding words' frequency from the V
    dictionary's values '''
    vocabKeysList = list(vocab.keys())
    vocabValuesList = list(vocab.values())
    setting.PRINTBOTH(f,"Total length=", len(track))
    count = 0
    setting.PRINTBOTH(f,"--------Normalising-------------")
    for row, columns in track.items():
        i = row
        count += 1
        for j in columns:
            freq_i = vocabValuesList[i]
            freq_j = vocabValuesList[j]
            weightMatrix[i][j] = weightMatrix[i][j] / min(freq_i,freq_j)
            if i==j:
                weightMatrix[i][j] = 1

    return weightMatrix


def createWeightMatrix(f,uniqueListOfTokenList, vocab,windowSize):
    '''This fuction takes in a corpus data in the form of list of sentences(lists), and V (Vocabulary) dictionary
    {word:count} and returns the W matrix which is a co-occurence matrix. Each cell of co-occurence matrix has the value of number of
    co-occerence of the words corresponding to the row and column of that cell, in the window of 5 words.'''

    track = {}
    targetWindow={}
    weightMatrix = np.zeros((len(vocab), len(vocab)), dtype='float32')
    vocabKeysList = list(vocab.keys())
    vocabValuesList = list(vocab.values())
    for tokensList in uniqueListOfTokenList:
        
        leftWindow = int(windowSize / 2)

        i = 0
        totalLen = len(tokensList)
        for target in tokensList:
            
            if totalLen<=windowSize:
                window=tokensList[0:totalLen]
            else:
                if i <= leftWindow:
                    window = tokensList[0:windowSize]
                elif i < totalLen-leftWindow:
                    if i - leftWindow >=0:
                        window = tokensList[i - leftWindow:i + leftWindow + 1]
                    else:
                        print("pass")
                else:
                    if totalLen - windowSize >=0:
                        window = tokensList[totalLen - windowSize:totalLen]
                    else:
                        print("pass")

            i += 1
            row = vocabKeysList.index(target)
            for word in window:
                if word != target:
                    column = vocabKeysList.index(word)
                    weightMatrix[row][column] += 1

                    try:
                        track[row].add(column)
                    except:
                        track[row] = {column}
                    try:
                        targetWindow[target].add(word)
                    except:
                        targetWindow[target] = {word}

    return weightMatrix, track


def main():
    # co-occur is created using all tokens
    logFile=os.path.basename(__file__).split(".")[0]+"_windowsize_{}.txt".format(setting.WINDOWSIZE)
    f=open(setting.LOGPATH+logFile, 'w')
    setting.PRINTBOTH(f,setting.TRAINVOCAB)
    if os.path.isfile(setting.WEIGHTMATRIXNORMALIZEDFILE):
        setting.PRINTBOTH(f,"removing ",setting.WEIGHTMATRIXNORMALIZEDFILE)
        os.remove(setting.WEIGHTMATRIXNORMALIZEDFILE)

    if os.path.isfile(setting.WEIGHTMATRIXFILE):
       setting.PRINTBOTH(f,"removing ",setting.WEIGHTMATRIXFILE)
       os.remove(setting.WEIGHTMATRIXFILE)


    vocab=pd.read_pickle(setting.TRAINVOCAB)
    uniqueListOfTokenList=create_vocab.getListOfTokenList(f,setting.TRAINDATASET,setting.TRAINDATA)
    setting.PRINTBOTH(f,"number of keys in vocab are ",len(vocab))
    flattenList = [x  for sublist in uniqueListOfTokenList for x in sublist]
    setting.PRINTBOTH(f," flattenlist : {} , set(flattenList) : {}".format(len(flattenList) , len(set(flattenList))))
    weightMatrix,track= createWeightMatrix(f,uniqueListOfTokenList,vocab,setting.WINDOWSIZE)
    weightMatrixFile = open(setting.WEIGHTMATRIXFILE, "wb")
    np.save(weightMatrixFile, weightMatrix)
    weightMatrixNormalizedArray = normaliseWeightMatrix(f,weightMatrix,vocab,track)
    weightMatrixNormalizedFile = open(setting.WEIGHTMATRIXNORMALIZEDFILE, "wb")
    # save array to the file
    setting.PRINTBOTH(f,"saving weightMatrixNormalizedFile : ",setting.WEIGHTMATRIXNORMALIZEDFILE)
    np.save(weightMatrixNormalizedFile, weightMatrixNormalizedArray)
    # close the file
    weightMatrixNormalizedFile.close()
    weightMatrixFile.close()
    f.close()

if __name__ == '__main__':
    main()