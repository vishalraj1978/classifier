# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:46:01 2023

@author: visha
"""
import os
import sys
PERSONAL=0
if PERSONAL:
    TRAINDATASET="D:/office desktop/AI/PROJECT/GlossBERT/Training_Corpora/SemCor/semcor_instance_compact_pos.csv"
else:
    TRAINDATASET="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Training_Corpora/SemCor/semcor_instance_compact_pos.csv"
MODELPATH="./model/"
PREDICTPATH="./predict/"
METRICPATH="./metric/"
#TRAIN_DATASET="semeval2013_instance_compact.csv"
TRAINDATA=MODELPATH+"train_data.pkl"
TRAINVOCAB=MODELPATH+"train_vocab.pkl"
WINDOWSIZE = 13



if not os.path.isdir(MODELPATH):
    os.mkdir(MODELPATH)
if not os.path.isdir(PREDICTPATH):
    os.mkdir(PREDICTPATH)
if not os.path.isdir(METRICPATH):
    os.mkdir(METRICPATH)
    

WEIGHTMATRIXNORMALIZEDFILE=MODELPATH+"Weight_Matrix_Normalized_File_window_{}.np".format(WINDOWSIZE)
WEIGHTMATRIXFILE=MODELPATH+"Weight_Matrix_File_window_{}.np".format(WINDOWSIZE)

LEMMAPOSCLUSTERFILE=MODELPATH+"lemma_pos_dict_indx_count_window_{}.pkl".format(WINDOWSIZE)

# Your model params:
CONTEXTWINDOW = 6
SGVALUE=0
NEGATIVES = 5
MINCOUNT = 5
EPOCHS = 20
MODELFILE=MODELPATH+"word2vec_window_{}_sg_{}.model".format(CONTEXTWINDOW,SGVALUE)



if PERSONAL:
    TESTDATASET_SENSEVAL3="D:/office desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/senseval3/senseval3_instance_compact_pos.csv"
    TESTDATASET_SENSEVAL2="D:/office desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/senseval2/senseval2_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2007="D:/office desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2007/semeval2007_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2013="D:/office desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2013/semeval2013_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2015="D:/office desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2015/semeval2015_instance_compact_pos.csv"
else:
    TESTDATASET_SENSEVAL3="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/senseval3/senseval3_instance_compact_pos.csv"
    TESTDATASET_SENSEVAL2="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/senseval2/senseval2_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2007="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2007/semeval2007_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2013="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2013/semeval2013_instance_compact_pos.csv"
    TESTDATASET_SEMEVAL2015="C:/Users/vishalr/OneDrive - STMicroelectronics/Desktop/AI/PROJECT/GlossBERT/Evaluation_Datasets/semeval2015/semeval2015_instance_compact_pos.csv"


TESTDATASET=[TESTDATASET_SENSEVAL2,TESTDATASET_SENSEVAL3,TESTDATASET_SEMEVAL2007,TESTDATASET_SEMEVAL2013,TESTDATASET_SEMEVAL2015]

#TESTDATASET=[TESTDATASET_SENSEVAL3]
LOGPATH="./log/"
if not os.path.isdir(LOGPATH):
    os.mkdir(LOGPATH)

def PRINTBOTH(file, *args):
    toprint = ' '.join([str(arg) for arg in args])
    print( toprint)
    file.write(toprint+"\n")

logFile=os.path.basename(__file__).split(".")[0]+".txt"

with open(LOGPATH+logFile, 'w') as f: PRINTBOTH(f,"vishal raj")

