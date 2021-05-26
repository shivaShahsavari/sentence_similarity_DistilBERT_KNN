import pandas as pd
import numpy as np
import os
import re
import json
import string
import nltk
import spacy
from sentence_transformers import SentenceTransformer
import math

def max_min_win(train):
    max_len=0
    max_str=''
    min_len=10
    min_str=''
    for i in range(train.shape[0]):
        if len(train['dataset_label'][i].split()) > max_len:
            max_len=len(train['dataset_label'][i].split())
            max_str=train['dataset_label'][i]
        elif len(train['dataset_label'][i].split()) < min_len:
            min_len=len(train['dataset_label'][i].split())
            min_str=train['dataset_label'][i]
    print('max length of label: ',max_len) #=> 17
    print('max_label: ',max_str) #=> "National Center for Science and Engineering Statistics Survey of Graduate Students and Postdoctorates in Science and Engineering"
    print('min length of label: ',min_len) #=> 1
    print('min_label: ',min_str) #=> ADNI
    return max_len,min_len

def main():

    train = pd.read_csv('train.csv')
    ############## Frirst EDA: identifying max/min words in dataset_label for CNN kernels ################ 
    print("Result of first EDA: ")
    max_win,min_win=max_min_win(train)

    ############## Second EDA: identifying top titles for Text summarization  ############################
    print("Result of second EDA: ")
    p_train=train[['Id','cleaned_label','text']]
    stat={}
    train_files_path='D:\\Self_Study\\Kaggle_Coleridge Initiative\\train'
    for i in range(p_train.shape[0]):
        print('file number : ',i)
        filename=p_train['Id'][i]
        json_path = os.path.join(train_files_path, (filename+'.json'))
        with open(json_path, 'r') as f:
            json_decode = json.load(f)
            ans=[(d['section_title'].lower(), d['text'].lower()) for d in json_decode]
            for j in range(len(ans)):
                result=ans[j][1].find(p_train['cleaned_label'][i])
                if (result != -1):
                    stat[ans[j][0]]= stat[ans[j][0]]+1 if ans[j][0] in stat.keys() else 1

    from collections import OrderedDict
    d_descending = OrderedDict(sorted(stat.items(),key=lambda kv: kv[1], reverse=True))
    print(len(d_descending))## total distinct titles of articles : 12828

    #######checking which articles not in top titles######
    title={}
    for i in range(p_train.shape[0]):
        print('file number : ',i)
        filename=p_train['Id'][i]
        json_path = os.path.join(train_files_path, (filename+'.json'))
        with open(json_path, 'r') as f:
            json_decode = json.load(f)
            ans=[(d['section_title'].lower(), d['text'].lower()) for d in json_decode]
            temp=[]
            for j in range(len(ans)):
                result=ans[j][1].find(p_train['cleaned_label'][i])
                if (result != -1):
                    temp.append(ans[j][0])
            common_title=list(set(temp).intersection(d_descending[:100]))
            if not common_title:
                title[filename]=1

    print('number of files not in top titles: ',len(title)) 
    ##number of files not in top 30 titles:  10064
    ##number of files not in top 40 titles:  9938

    ############## Third EDA: number of articles not ivolved for Coine similarity of vector sentences transformed by Bert algorithm ########
    print("Result of third EDA: ")
    def cosine_similarity(v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

    p_train2=train[['Id','cleaned_label','dataset_label','text']]

    art_dic=pd.DataFrame(columns=['Id','sent_vec','clasS_vec','sim_prc'])
    art_meta=[]

    for i in range(p_train2.shape[0]):
        print('file number : ',i)
        filename=p_train2['Id'][i]
        json_path = os.path.join(train_files_path, (filename+'.json'))
        contents = []
        with open(json_path, 'r') as f:
            json_decode = json.load(f)
            for data in json_decode:
                contents.append(data.get('text'))

        all_contents = ' '.join(contents)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        all_sent=tokenizer.tokenize(all_contents)
        print ('number of sentences in article: ',len(all_sent))

    #######Bert transform
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        query = p_train2['dataset_label'][i]
        query_vec = sbert_model.encode([query])[0]

        for sent in all_sent:
            if sent.find(query) != -1 :
                #print('shape of vector: ',sbert_model.encode([all_sent])[0].shape)## (768,)
                sim = cosine_similarity(query_vec,sbert_model.encode([sent])[0])
                if sim < 0.4 :
                    art_meta.append(p_train2['Id'][i])
        
    print('number of articles contains key and sim <40 : ',len(np.unique(art_meta)))
    #number of articles contains key and sim <40 :  5519

if __name__ == "__main__":
    main()