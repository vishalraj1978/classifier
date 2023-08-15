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
import sys
import setting

logFile=os.path.basename(__file__).split(".")[0]+"_windowsize_{}.txt".format(setting.WINDOWSIZE)
f=open(setting.LOGPATH+logFile, 'w')

vocab=pd.read_pickle(setting.TRAINVOCAB)
vocabKeysList = list(vocab.keys())

DATASET=setting.TRAINDATASET
train_df = pd.read_pickle(setting.TRAINDATA)
if  'token_list' in train_df.columns and isinstance(train_df.loc[0,'token_list'], type("str")):
    train_df['token_list'] = train_df['token_list'].apply(lambda token_list: ast.literal_eval(token_list))
if train_df.isnull().values.any():
    setting.PRINTBOTH(f,"some column of train_df is null")
if train_df[train_df['lemma_instance'].isnull()].index.tolist():
    train_df['lemma_instance'].fillna('null', inplace=True)
if train_df[train_df['target_lemma'].isnull()].index.tolist():
    train_df['target_lemma'].fillna('null', inplace=True)
if train_df.isnull().values.any():
    setting.PRINTBOTH(f,"some column of train_df is null")


weightMatrixNormalizedFile = open(setting.WEIGHTMATRIXNORMALIZEDFILE,"rb")
weightMatrixNormalizedArray = np.load(weightMatrixNormalizedFile)
weightMatrixNormalizedFile.close()
weight_matrix_df=pd.DataFrame(weightMatrixNormalizedArray)

pos_df=pd.DataFrame()
new_df = pd.DataFrame()
lemma_clusters={}
count=0


setting.PRINTBOTH(f,"Total number of lemmas are : {}".format(len(train_df.target_lemma.unique())))
for lemma in train_df.target_lemma.unique(): 
    #if lemma!="person":
    #    continue
    count+=1
    #df = df[df.target_pos == "NOUN"]
    lemma_df = train_df[train_df.target_lemma == lemma]
    setting.PRINTBOTH(f,"############################ count : {}, lemma:{} ####################################".format(count,lemma))
    setting.PRINTBOTH(f,"len(lemma_df) total number of instances : {}".format(len(lemma_df)))
    setting.PRINTBOTH(f,"len(lemma_df.sense_key.unique()) : {},lemma_df.sense_key.unique() : {}".format(len(lemma_df.sense_key.unique()),lemma_df.sense_key.unique()))
    pos_clusters={}
   
    for pos in lemma_df.target_pos.unique():
        pos_df = lemma_df[lemma_df.target_pos == pos]
        setting.PRINTBOTH(f,"\tpos: {}".format(pos))
        setting.PRINTBOTH(f,"\ttotal number of instances (len(pos_df)) :{}  ".format(len(pos_df)))
        setting.PRINTBOTH(f,"\tlen(pos_df.sense_key.unique()) : {},pos_df.sense_key.unique() : {}".format(len(pos_df.sense_key.unique()),pos_df.sense_key.unique()))
    

        sense_keys_df={}
        bow_senses_token_list={}
        bow_senses_flatten_list={}
        bow_senses_indexs={}
        bow_senses_flatten_indexs={}
        bow_senses_flatten_indexs_dict={}
        #weight_matrix_df=pd.DataFrame(weightMatrixNormalizedArray)

        for sense in pos_df.sense_key.unique():
            if sense == 'long%3:00:01::':
                setting.PRINTBOTH(f,"sense")
            sense_keys_df[sense]=pos_df[pos_df.sense_key==sense]
            def removing(row):
                val = [x for x in row['token_list'] if x != row['target_lemma']+"_"+row['target_pos']]
                return val
            #setting.PRINTBOTH(f,sense_keys_df[sense].apply(removing,axis=1).tolist())
            wt_i=weightMatrixNormalizedArray[vocabKeysList.index(lemma+"_"+pos)]
            wt_i_idxs = np.where(wt_i > 0)[0]
            bow_senses_token_list[sense]=sense_keys_df[sense].apply(removing,axis=1).tolist()
            bow_senses_flatten_list[sense] = [x  for sublist in bow_senses_token_list[sense] for x in sublist]
            bow_senses_flatten_indexs[sense] = [vocabKeysList.index(x)  for x in bow_senses_flatten_list[sense]]
            #temp_df["sum"]=temp_df.sum(axis=1)
            from collections import Counter
            bow_senses_flatten_indexs_dict[sense]=dict(Counter(bow_senses_flatten_indexs[sense]))
            bow_senses_flatten_indexs_dict[sense] = {k: bow_senses_flatten_indexs_dict[sense][k] for k in set(wt_i_idxs) & set(bow_senses_flatten_indexs_dict[sense].keys())}
            bow_senses_flatten_indexs[sense] = set(bow_senses_flatten_indexs[sense])
            
        
        pos_clusters[pos]=bow_senses_flatten_indexs_dict
    
    lemma_clusters[lemma]=pos_clusters

    new_df = new_df.append(lemma_df)
    #if count==1:
       #break
       #setting.PRINTBOTH(f,"")

f.close()
with open (setting.LEMMAPOSCLUSTERFILE, 'wb') as lemmaClustersFile:
    pickle.dump(lemma_clusters, lemmaClustersFile)
