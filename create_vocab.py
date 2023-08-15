# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:32:56 2023

@author: visha
"""
import pickle
import subprocess
import os
from pandas import read_csv   
import sys 
import ast
import pandas as pd
import re
import nltk
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim.downloader as api
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import string
import setting


import gensim
from gensim.models.callbacks import CallbackAny2Vec



stopWordList = list(stopwords.words('english'))



class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1

loss_logger = LossLogger()
"""mod = gensim.models.word2vec.Word2Vec(sentences=sentences,
                                      sg=1,
                                      window=CONTEXTWINDOW,
                                      negative=NEGATIVES,
                                      min_count=MIN_COUNT,
                                      callbacks=[loss_logger],
                                      compute_loss=True,
                                      epochs=EPOCHS)"""



def createVocab(corpus):
    '''This function takes an input as a corpus in the form os list of sentences and returns a dictionary V,
    with keys as words in the corpus and values as their frequency in the corpus '''
    vocab = {}
    for sentence in corpus:
        for word in sentence:
            if word in stopWordList:
                setting.PRINTBOTH(f,word)
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
    return vocab

def getVocab(f,trainDataset,uniqueListOfTokenList,trainVocab):
    setting.PRINTBOTH(f,"Creating new vocabulory {}".format(trainVocab)) 
    if not uniqueListOfTokenList:
        uniqueListOfTokenList=getListOfTokenList(trainDataset)
    vocab=createVocab(uniqueListOfTokenList)
    setting.PRINTBOTH(f,"saving vocabulory : ",'model_vocab.pkl')
    with open (trainVocab, 'wb') as vocabFile:
        pickle.dump(vocab, vocabFile)     
    setting.PRINTBOTH(f,"number of keys in vocab are ",len(vocab))
    return vocab

def getListOfTokenList(f,tainDataset,trainData):
    #currentPath = subprocess.check_output("pwd", shell=True, universal_newlines=True)
    if os.path.isfile(tainDataset):
        df = read_csv(tainDataset, sep="\t", encoding="utf-8")
        setting.PRINTBOTH(f,df.columns)
    else:
        setting.PRINTBOTH(f,"{} does not exist in {}".format(tainDataset,"."))
        setting.PRINTBOTH(f,"Please copy atleast one of them")
        sys.exit()
    
    if  'token_list' in df.columns and isinstance(df.loc[0,'token_list'], type("str")):
        df['token_list'] = df['token_list'].apply(lambda token_list: ast.literal_eval(token_list))
        #trainData.to_pickle('train_model.pkl')
        setting.PRINTBOTH(f,"Saving token list {}".format(trainData))
    df.to_pickle(trainData)   
    setting.PRINTBOTH(f,"len(df['token_list'].tolist())",len(df['token_list'].tolist()))
    listOfTokenList=df['token_list'].tolist()
    prevTokenList=[]
    uniqueListOfTokenList=[]
    for tokenList in listOfTokenList:
        if prevTokenList==tokenList:
            continue
        uniqueListOfTokenList.append(tokenList)
        prevTokenList=tokenList
    setting.PRINTBOTH(f,"number of uniqueListOfTokenList in dataset are ",len(uniqueListOfTokenList))
    flattenList = [x  for sublist in uniqueListOfTokenList for x in sublist]
    setting.PRINTBOTH(f," flattenlist : {} , set(flattenList) : {}".format(len(flattenList) , len(set(flattenList))))
    return uniqueListOfTokenList

def getModel(f,uniqueListOfTokenList,contextWindow,sgValue):
    if not uniqueListOfTokenList:
        uniqueListOfTokenList=getListOfTokenList(f,setting.TRAINDATASET)
    #model=Word2Vec(uniqueListOfTokenList, vector_size=300, window=4, min_count=1, hs=0, negative=5, sg=1 ,epochs=10, workers=4,callbacks=[loss_logger],compute_loss=True, )
    #model=Word2Vec(uniqueListOfTokenList, vector_size=300, window=9, min_count=1, hs=0, negative=5, sg=1 ,epochs=10, workers=4,callbacks=[loss_logger],compute_loss=True, )
    model= Word2Vec(uniqueListOfTokenList,vector_size=300,min_count=1,window=contextWindow,hs=0,negative=5,sg=sgValue,workers=4,callbacks=[loss_logger],compute_loss=True, )
    
    
    return model
def main():
    logFile=os.path.basename(__file__).split(".")[0]+"_windowsize_{}.txt".format(setting.WINDOWSIZE)
    f=open(setting.LOGPATH+logFile, 'w')
    if os.path.isfile(setting.TRAINDATA):
        setting.PRINTBOTH(f,"removing ",setting.TRAINDATA)
        os.remove(setting.TRAINDATA)
    if os.path.isfile(setting.TRAINVOCAB):
        setting.PRINTBOTH(f,"removing ",setting.TRAINVOCAB)
        os.remove(setting.TRAINVOCAB)
    if os.path.isfile(setting.MODELFILE):
        setting.PRINTBOTH(f,"removing ",setting.MODELFILE)
        os.remove(setting.MODELFILE)
    uniqueListOfTokenList=getListOfTokenList(f,setting.TRAINDATASET,setting.TRAINDATA)
    vocab=getVocab(f,setting.TRAINDATASET,uniqueListOfTokenList,setting.TRAINVOCAB)    
    model=getModel(f,uniqueListOfTokenList,setting.CONTEXTWINDOW,setting.SGVALUE)
    setting.PRINTBOTH(f,"saving model : ",setting.MODELFILE)
    model.save(setting.MODELFILE)
    setting.PRINTBOTH(f,"number of keys in word2vec model are ",len(model.wv.key_to_index))
    setting.PRINTBOTH(f,"number of keys in vocab are ",len(vocab))
    setting.PRINTBOTH(f,"number of  uniqueListOfTokenList in dataset are ",len(uniqueListOfTokenList))
    setting.PRINTBOTH(f,"========checking by loading saved models now ========")
    vocab=pd.read_pickle(setting.TRAINVOCAB)
    model = Word2Vec.load(setting.MODELFILE)
    setting.PRINTBOTH(f,"number of keys in word2vec model are ",len(model.wv.key_to_index))
    setting.PRINTBOTH(f,"number of keys in vocab are ",len(vocab))
    df = pd.read_pickle(setting.TRAINDATA)
    setting.PRINTBOTH(f,"len(df['token_list'].tolist())",len(df['token_list'].tolist()))
    listOfTokenList=df['token_list'].tolist()
    prevTokenList=[]
    uniqueListOfTokenList=[]
    for tokenList in listOfTokenList:
        if prevTokenList==tokenList:
            continue
        uniqueListOfTokenList.append(tokenList)
        prevTokenList=tokenList
    setting.PRINTBOTH(f,"number of uniqueListOfTokenList in dataset are ",len(uniqueListOfTokenList))
    flattenList = [x  for sublist in uniqueListOfTokenList for x in sublist]
    setting.PRINTBOTH(f," flattenlist : {} , set(flattenList) : {}".format(len(flattenList) , len(set(flattenList))))
    setting.PRINTBOTH(f,"\small_NOUN")
    setting.PRINTBOTH(f,model.wv.most_similar(positive=["small_NOUN"], topn=10))    
    setting.PRINTBOTH(f,"\long_ADJ")
    setting.PRINTBOTH(f,model.wv.most_similar(positive=["long_ADJ"], topn=10))
    setting.PRINTBOTH(f,"\art_NOUN")
    setting.PRINTBOTH(f,model.wv.most_similar(positive=["art_NOUN"], topn=10))
    setting.PRINTBOTH(f,"\nvending_machine_NOUN")
    setting.PRINTBOTH(f,model.wv.most_similar(positive=["vending_machine_NOUN"], topn=10))
    f.close()
    #model.wv.most_similar(positive=["be"], topn=10)
if __name__ == '__main__':
    main()