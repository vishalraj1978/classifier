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

logFile=os.path.basename(__file__).split(".")[0]+"_windowsize_{}.txt".format(setting.WINDOWSIZE)
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



new_df=train_df   
predicted_list=[]
mfs_predicted_list=[]
count=0
entries=[]
senseKeyNotPresent=0
notPredictedCount=0
notPresentTarget=0
notPresentLemmaPosCluster=0
predictSenseKeyWithZero=0
metricDF = pd.DataFrame(index =['wm_macro', 'wm_micro', 'wm_match','wm_not_match','mfs_macro', 'mfs_micro','mfs_match','mfs_not_match']) 
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
        notPresentTarget+=1
        #setting.PRINTBOTH(f,"{}:{} is not present in vocab".format(target_indx,target_token))
        continue 
    try:
        sense_clusters=lemma_clusters[lemma][pos]
    except:
        notPresentLemmaPosCluster+=1
        #setting.PRINTBOTH(f,"ERROR : lemma : {},pos : {} is not in pos_clusters".format(lemma,pos,row))
        continue
    
    predict=train_df[ ( train_df.target_lemma == lemma)]['sense_key'].value_counts().index[0]
    if row.sense_key not in train_df[ (train_df.target_pos == pos) & (train_df.target_lemma == lemma)]['sense_key'].unique():
        #setting.PRINTBOTH(f," {} does not exist in train dataframe ".format(row.sense_key))
        senseKeyNotPresent+=1
        continue
    if predict=="":
        notPredictedCount+=1
        continue
        setting.PRINTBOTH(f,"ERROR : predict_sense_key is '' and row is {} ".format(row))
        predict=train_df[ (train_df.target_pos == pos) & (train_df.target_lemma == lemma)]['sense_key'].value_counts().index[0]
        setting.PRINTBOTH(f,"Setting the predict to most occured sense in training data")
        #continue    
    """if row.sense_key != predict:
        setting.PRINTBOTH(f,"break")
    setting.PRINTBOTH(f,"{},   True : {}, Predict: {}".format(row.sense_key==predict,row.sense_key,predict))"""
    #setting.PRINTBOTH(f, 'true : {} , predict : {}'.format(row.sense_key,predict))
    entries.append(row)
    predicted_list.append(predict)
    mfs_predict=train_df[(train_df.target_lemma == lemma)]['sense_key'].value_counts().index[0]
    mfs_predicted_list.append(mfs_predict)
    #if(len(predicted_list)==100):
        #break
setting.PRINTBOTH(f,"\n EVALUATION OF LEMMA CLUSTERS for train dataset {}\n")
setting.PRINTBOTH(f,"senseKeyNotPresent : {}, notPredictedCount : {} ".format(senseKeyNotPresent, notPredictedCount))
setting.PRINTBOTH(f,"notPresentTarget : {}, notPresentLemmaPosCluster : {} ".format(notPresentTarget, notPresentLemmaPosCluster))    
setting.PRINTBOTH(f,"predictSenseKeyWithZero : {}".format(predictSenseKeyWithZero, ))    
test_df=pd.DataFrame(entries)
setting.PRINTBOTH(f,"len(test_df):{}, len(predicted_list):{}".format(len(test_df),len(predicted_list)))
setting.PRINTBOTH(f,"len(new_df):{}, len(test_df):{}".format(len(new_df),len(test_df)))
test_df["predict_sense_key"]=predicted_list 
test_df["mfs_predict_sense_key"]=mfs_predicted_list 
from sklearn.metrics import f1_score

test_df.to_csv(setting.PREDICTPATH+os.path.basename(__file__).split(".")[0]+"_window_{}_predict.csv".format(setting.WINDOWSIZE),sep="\t")   
posMetricDF = pd.DataFrame(index =['wm_pos_macro', 'wm_pos_micro', 'pos_match','pos_not_match','mfs_pos_macro', 'mfs_pos_micro','mfs_pos_match','mfs_pos_not_match']) 
for pos in test_df.target_pos.unique():
    pos_df = test_df[test_df.target_pos == pos]
    
    wm_pos_macro=f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='macro')
    wm_pos_micro=f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='micro')
    mfs_pos_macro=f1_score(pos_df["sense_key"], pos_df["mfs_predict_sense_key"], average='macro')
    mfs_pos_micro=f1_score(pos_df["sense_key"], pos_df["mfs_predict_sense_key"], average='micro') 
    pos_series=np.where(pos_df['sense_key'] == pos_df['predict_sense_key'], True, False)
    mfs_pos_series=np.where(pos_df['sense_key'] == pos_df['mfs_predict_sense_key'], True, False)
    pos_match=len(pos_series[pos_series==True])
    pos_not_match=len(pos_series[pos_series==False])
    mfs_pos_match=len(mfs_pos_series[mfs_pos_series==True])
    mfs_pos_not_match=len(mfs_pos_series[mfs_pos_series==False])        
    posMetricDF[pos]=[wm_pos_macro,wm_pos_micro,pos_match,pos_not_match,mfs_pos_macro,mfs_pos_micro,mfs_pos_match,mfs_pos_not_match]
    
    
    setting.PRINTBOTH(f,"$$$$$$$$$$$$$$$$$$$$$$$ pos:{} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$".format(pos))
    setting.PRINTBOTH(f,"len(pos_df) total number of instances : {}".format(len(pos_df)))
    setting.PRINTBOTH(f,"\t total number of sense :{} ".format(len(pos_df.sense_key.unique())))
    setting.PRINTBOTH(f,"\t SCORE(Macro) : ",f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='macro'))
    setting.PRINTBOTH(f,"\t SCORE(Micro) : ",f1_score(pos_df["sense_key"], pos_df["predict_sense_key"], average='micro'))

    setting.PRINTBOTH(f,"\t Count of Total :{}, Count of matches :{}, Count of non matches :{}".format(len(pos_series),len(pos_series[pos_series==True]),len(pos_series[pos_series==False]))) 

posMetricDF.to_csv(setting.METRICPATH+os.path.basename(__file__).split(".")[0]+"_{}_window_{}_metric.csv".format(os.path.basename(setting.TRAINDATASET).split(".")[0].split("_")[0],setting.WINDOWSIZE),sep="\t") 
wm_macro=f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='macro')
wm_micro=f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='micro')
mfs_macro=f1_score(test_df["sense_key"], test_df["mfs_predict_sense_key"], average='macro')
mfs_micro=f1_score(test_df["sense_key"], test_df["mfs_predict_sense_key"], average='micro') 
match_series=np.where(test_df['sense_key'] == test_df['predict_sense_key'], True, False)
mfs_match_series=np.where(test_df['sense_key'] == test_df['mfs_predict_sense_key'], True, False)
wm_match=len(match_series[match_series==True])
wm_not_match=len(match_series[match_series==False])
mfs_match=len(mfs_match_series[mfs_match_series==True])
mfs_not_match=len(mfs_match_series[mfs_match_series==False])
metricDF[os.path.basename(setting.TRAINDATA).split(".")[0].split("_")[0]]=[wm_macro,wm_micro,wm_match,wm_not_match,mfs_macro,mfs_micro,mfs_match,mfs_not_match]
setting.PRINTBOTH(f,"############################Total score####################################")    
setting.PRINTBOTH(f,"SCORE(Macro) : ",f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='macro'))
setting.PRINTBOTH(f,"SCORE(Micro) : ",f1_score(test_df["sense_key"], test_df["predict_sense_key"], average='micro'))

setting.PRINTBOTH(f,"Count of Total :{}, Count of matches :{}, Count of non matches :{}".format(len(match_series),len(match_series[match_series==True]),len(match_series[match_series==False]))) 
metricDF.to_csv(setting.METRICPATH+os.path.basename(__file__).split(".")[0]+"_window_{}_metric.csv".format(setting.WINDOWSIZE),sep="\t") 
#df1 = read_csv("predict_wm_test_data_semeval2015_metric.csv", sep="\t", encoding="utf-8",index_col=0)
#sys.stdout.close()
f.close()



