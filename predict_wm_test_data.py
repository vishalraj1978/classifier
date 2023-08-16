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

setting.PRINTBOTH(f,"Loading existing  : ",setting.WEIGHTMATRIXNORMALIZEDFILE)
weightMatrixNormalizedFile = open(setting.WEIGHTMATRIXNORMALIZEDFILE,"rb")
weightMatrixNormalizedArray = np.load(weightMatrixNormalizedFile)
weightMatrixNormalizedFile.close()
weightMatrixNormalizedArray1=np.where(weightMatrixNormalizedArray>1, 1, weightMatrixNormalizedArray)
weight_matrix_df=pd.DataFrame(weightMatrixNormalizedArray1)  
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


    new_df=df   
    predicted_list=[]
    count=0
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
        
        for context_token in row.token_list:
            if context_token==target_token:
                continue
            try:
                context_indx=vocabKeysList.index(context_token)
            except:
                #setting.PRINTBOTH(f,"ERROR : {} is not in vocab during prediction".format(context_token))
                continue 
            max_affinity=0
            real_sense=""
            affinity_list=[]
            min_val = min([len(cluster) for sense,cluster in pos_clusters.items()])
            for sense,cluster in pos_clusters.items():
                #min_val = min([len(cluster) for sense,cluster in pos_clusters.items()])
                """cluster_nonzero_indxs=[x for x in cluster if weight_matrix_df.iloc[token_index,x]!=0]
                affinity=weight_matrix_df.iloc[token_index,cluster_nonzero_indxs].sum()
                len_cluster_nonzero_indxs=len(cluster_nonzero_indxs)"""
    
                 
                affinity=0
                affinity_list=[weight_matrix_df.iloc[context_indx,key] for key,value in cluster.items()]
                #affinity_list=[weight_matrix_df.iloc[context_indx,vocabKeysList.index(key)] for key,value in cluster.items()]
                
                """affinity_dict={key:weight_matrix_df.iloc[context_indx,key] for key,value in cluster.items()}
                maxKey = max(affinity_dict, key=affinity_dict.get)
                while min_val:
                    maxKey = max(affinity_dict, key=affinity_dict.get)
                    removed_value = affinity_dict.pop(maxKey)
                    if min_val > cluster[maxKey]:
                        affinity+=cluster[maxKey]*removed_value
                        min_val-=cluster[maxKey]
                    else:
                        affinity+=min_val*removed_value
                        min_val-=min_val"""
                #div=len(train_df[ (train_df.sense_key == sense)])
                #div=len(cluster)
                #if affinity != 0:
                    #affinity/=div
                
                affinity_list.sort(reverse=True)
                
                affinity=sum(affinity_list[0:min_val])
                affinity/=(min_val)
                try:
                    predict_sense_key[sense]+=affinity
                except:
                    predict_sense_key[sense]=affinity
                """if(affinity>max_affinity):
                    real_sense=sense
                    max_affinity=affinity
                if max_affinity==0:
                    if context_indx in cluster:
                        real_sense=sense"""
            """try:
                predict_sense_key[real_sense]+=max_affinity
            except:
                predict_sense_key[real_sense]=max_affinity"""
        #setting.PRINTBOTH(f,"Number of clusters in this {}-{}: {}".format(lemma,pos,len(pos_clusters)))
        predict=max(predict_sense_key, key=predict_sense_key.get, default="")
        if row.sense_key not in train_df[ (train_df.target_pos == pos) & (train_df.target_lemma == lemma)]['sense_key'].unique():
            senseKeyNotPresent+=1
            #setting.PRINTBOTH(f," {} does not exist in train dataframe ".format(row.sense_key))
            continue
        #if row.sense_key != predict:
            #setting.PRINTBOTH(f,"break")
            #setting.PRINTBOTH(f,"{},   True : {}, Predict: {}".format(row.sense_key==predict,row.sense_key,predict))
        if predict=="":
            notPredictedCount+=1
            continue
            #setting.PRINTBOTH(f,"ERROR : predict_sense_key is '' and row is {} ".format(row))
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