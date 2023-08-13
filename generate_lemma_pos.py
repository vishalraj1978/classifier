import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
import string
import gensim
stopWordList = list(stopwords.words('english'))

def generate(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()
    org_sentences=[]
    sentences = []
    all_token_list = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            #sent=sent.tolower()
            org_sentence=[]
            sentence = []
            token_list = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                if token.text=="U.N.":
                    print("stop")
                if token.text=="Laurence M . Klauber" or token.text== "Maude" or token.text== "Professor Fredrik Bo":
                    print("i m here")
                org_sentence.append(token.text)
                assert token.tag == 'wf' or token.tag == 'instance'
                
                
                s1=token.text.lower()
                s1 = s1.translate(str.maketrans('', '', string.punctuation))
                s1 = " ".join(s1.split())
                
                s2=gensim.utils.simple_preprocess(token.text)
                s2 = "_".join(s2)
                
                if s1 in stopWordList or s2 in stopWordList: 
                    continue
                if not s1 or not s2:
                    continue 
                
                if s1 !=s2:
                    print("ERROR:token.text:{}\t\t\ts1:{}\t\t\ts2:{} ".format(token.text,s1,s2))

                token.text=s2
                
                

                if token.tag == 'wf':
                    #continue                    
                    for i in token.text.split():
                        sentence.append(i)
                        #token_list.append(i)
                        token_list.append(token.attrib['lemma']+"_"+token.attrib['pos'])
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    if token.attrib['id']=="d000.s027.t007":
                        print("pass")
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    #token_list.append(token.text)
                    token_list.append(token.attrib['lemma']+"_"+token.attrib['pos'])
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            if len(sentence)!=len(pos) or len(target_index_start)!=len(target_index_end)\
                or len(pos)!=len(lemma) or len(pos)!=len(target) or len(pos)!=len(token_list):
                    print("ERRRRRROR")
            sentences.append(sentence)
            org_sentences.append(org_sentence)
            all_token_list.append(token_list)
            poss.append(pos) # two pos if token is of length 2
            targets.append(target) # two target id if token is of length 2
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma) # two lemma if token is of length 2
    if (len(sentences)==len(org_sentences) and  len(sentences)==len(all_token_list) \
        and len(sentences)==len(poss) and  len(sentences)==len(targets)\
        and len(sentences)==len(targets_index_start) and  len(sentences)==len(targets_index_end)\
        and len(sentences)==len(lemmas)):
        print(len(sentences),len(org_sentences))
    else:
        print("ERROR",len(sentences),len(org_sentences),len(all_token_list), \
        len(poss),len(targets),len(targets_index_start),len(targets_index_end)\
        ,len(sentences),len(lemmas))

    
    gold_keys = []
    gold_keys_dict={}
    with open(file_name[:-len('.data.xml')] + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys_dict[key[0]]=key[1]
            gold_keys.append(key[1])
            key = m.readline().strip().split()
    print("len(gold_keys_dict)",len(gold_keys_dict))

    output_file = file_name[:-len('.data.xml')] + '_instance_compact_pos.csv'
    used_gold_keys_dict={}
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('org_sentence\tsentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\tlemma_instance\ttoken_list\n')
        num = 0
        unused_sentences=0
        used_sentences=0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                org_sentence=' '.join(org_sentences[i])
                if (len(sentences[i])<=1 or len(all_token_list[i])<=1):
                    print(org_sentence,"#####",sentences[i],"#####",all_token_list[i])
                    unused_sentences+=1    #sentences which are not used for multiple targets
                    continue
                else:
                    used_sentences+=1
                sentence = ' '.join(sentences[i])
                token_list = all_token_list[i]
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                lemma_instance=' '.join(sentence.split()[target_start:target_end])
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                #sense_key = gold_keys[num]
                sense_key = gold_keys_dict[target_id]
                used_gold_keys_dict[target_id]=gold_keys_dict[target_id]
                num += 1   # sentences which are used for multiple targets, one sentence can have multiple targets
                g.write('\t'.join((org_sentence,sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key,lemma_instance,str(token_list))))
                g.write('\n')
    unused_gold_keys_dict=dict(set(gold_keys_dict.items()) - set(used_gold_keys_dict.items()))

    print(" used_sentences:{} unused_sentences:{}".format(used_sentences,unused_sentences))
    #print("num:{},unused_num:{},len(gold_keys_dict):{}".format(num,unused_num,len(gold_keys_dict)))
    #print("size of output  file : {}, length of gold keys dictionary : {}".format(num,len(gold_keys_dict)))
    #print("size of used gold keys  : {}, size of unused gold keys : {}".format(len(used_gold_keys_dict),len(unused_gold_keys_dict)))
if __name__ == "__main__":
    eval_dataset = ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL']
    # train_dataset = ['SemCor', 'SemCor+OMSTI']
    train_dataset = ['SemCor']

    file_name = []
    for dataset in eval_dataset:
        file_name.append('../Evaluation_Datasets/' + dataset + '/' + dataset + '.data.xml')
    for dataset in train_dataset:
        file_name.append('../Training_Corpora/' + dataset + '/' + dataset.lower() + '.data.xml')

    for file in file_name:
        print(file)
        generate(file)