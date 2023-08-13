# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:05:03 2023

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

logFile=os.path.basename(__file__).split(".")[0]+".txt"
f=open(setting.LOGPATH+logFile, 'w')


vocab=pd.read_pickle(setting.TRAINVOCAB)
vocabKeysList = list(vocab.keys())

train_df=pd.read_pickle(setting.TRAINDATA)
if train_df.isnull().values.any():
    setting.PRINTBOTH(f,"some column of train_df is null")
if train_df[train_df['lemma_instance'].isnull()].index.tolist():
    train_df['lemma_instance'].fillna('null', inplace=True)
if train_df[train_df['target_lemma'].isnull()].index.tolist():
    train_df['target_lemma'].fillna('null', inplace=True)
if train_df.isnull().values.any():
    setting.PRINTBOTH(f,"some column of train_df is null")

setting.PRINTBOTH(f,"Loading existing  : ",setting.LEMMAPOSCLUSTERFILE)
lemma_clusters=pd.read_pickle(setting.LEMMAPOSCLUSTERFILE)  

for testDataset in setting.TESTDATASET:
    if os.path.isfile(testDataset):
        df = read_csv(testDataset, sep="\t", encoding="utf-8")
        setting.PRINTBOTH(f,df.columns)
    else:
        setting.PRINTBOTH(f,"{} does not exist in {}".format(testDataset,"."))
        setting.PRINTBOTH(f,"Please copy atleast one of them")
        sys.exit()
    
    if  'token_list' in df.columns and isinstance(df.loc[0,'token_list'], type("str")):
        df['token_list'] = df['token_list'].apply(lambda token_list: ast.literal_eval(token_list))
        #trainData.to_pickle('train_model.pkl')




new_df=train_df    
predicted_list=[]
entries=[]

senseKeyNotPresent=0
notPredictedCount=0
notPresentTarget=0
notPresentLemmaPosCluster=0
for index,row in new_df.iterrows():
    #setting.PRINTBOTH(f,index,row)
    predict_sense_key={}
    lemma=row.target_lemma
    lemma_instance=row.lemma_instance
    pos=row.target_pos
    target_token=lemma+"_"+pos
    try:
        target_indx=vocabKeysList.index(target_token)
    except:
        #setting.PRINTBOTH(f,"{}:{} is not present in vocab".format(target_indx,target_token))
        notPresentTarget+=1
        continue 
    try:
        pos_clusters=lemma_clusters[lemma][pos]
    except:
        notPresentLemmaPosCluster+=1
        #setting.PRINTBOTH(f,"ERROR : lemma : {},pos : {} is not in pos_clusters".format(lemma,pos,row))
        continue
    
    if row.sense_key not in train_df[ (train_df.target_pos == pos) & (train_df.target_lemma == lemma)]['sense_key'].unique():
        senseKeyNotPresent+=1
        #setting.PRINTBOTH(f," {} does not exist in train dataframe ".format(row.sense_key))
        continue
    #if row.sense_key != predict:
        #setting.PRINTBOTH(f,"break")
        #setting.PRINTBOTH(f,"{},   True : {}, Predict: {}".format(row.sense_key==predict,row.sense_key,predict))
    predict=train_df[ (train_df.target_pos == pos) & (train_df.target_lemma == lemma)]['sense_key'].value_counts().index[0]
        #setting.PRINTBOTH(f,"Setting the predict to most occured sense in training data")
        #continue
    #setting.PRINTBOTH(f, 'true : {} , predict : {}'.format(row.sense_key,predict))
    entries.append(row)
    predicted_list.append(predict)
    #if(len(predicted_list)==100):
        #break
setting.PRINTBOTH(f,"senseKeyNotPresent : {}, notPredictedCount : {} ".format(senseKeyNotPresent, notPredictedCount))
setting.PRINTBOTH(f,"notPresentTarget : {}, notPresentLemmaPosCluster : {} ".format(notPresentTarget, notPresentLemmaPosCluster))    
test_df=pd.DataFrame(entries)
setting.PRINTBOTH(f,"len(test_df):{}, len(predicted_list):{}".format(len(test_df),len(predicted_list)))
setting.PRINTBOTH(f,"len(new_df):{}, len(test_df):{}".format(len(new_df),len(test_df)))
test_df["predict_sense_key"]=predicted_list    
from sklearn.metrics import f1_score

test_df.to_csv(setting.PREDICTPATH+os.path.basename(__file__).split(".")[0]+"_predict.csv",sep="\t")
setting.PRINTBOTH(f,"\n EVALUATION OF LEMMA CLUSTERS for test dataset {}\n".format(testDataset))
for pos in test_df.target_pos.unique():
    pos_df = test_df[test_df.target_pos == pos]
    setting.PRINTBOTH(f,"$$$$$$$$$$$$$$$$$$$$$$$ pos:{} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".format(pos))
    setting.PRINTBOTH(f,"len(pos_df) total number of instances : {}".format(len(pos_df)))
    setting.PRINTBOTH(f,"\t total number of sense :{} ".format(len(pos_df.sense_key.unique())))
    setting.PRINTBOTH(f,"\t SCORE(Macro) : ",f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='macro'))
    setting.PRINTBOTH(f,"\t SCORE(Micro) : ",f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='micro'))
    pos_series=np.where(pos_df['sense_key'] == pos_df['predict_sense_key'], True, False)
    setting.PRINTBOTH(f,"\t Count of Total :{}, Count of matches :{}, Count of non matches :{}".format(len(pos_series),len(pos_series[pos_series==True]),len(pos_series[pos_series==False]))) 

 

setting.PRINTBOTH(f,"############################Total score####################################")    
setting.PRINTBOTH(f,"SCORE(Macro) : ",f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='macro'))
setting.PRINTBOTH(f,"SCORE(Micro) : ",f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='micro'))
match_series=np.where(test_df['sense_key'] == test_df['predict_sense_key'], True, False)
setting.PRINTBOTH(f,"Count of Total :{}, Count of matches :{}, Count of non matches :{}".format(len(match_series),len(match_series[match_series==True]),len(match_series[match_series==False]))) 
#sys.stdout.close()
f.close()