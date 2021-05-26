# List all installed packages and package versions
#!pip install sentence_transformers


import numpy as np
import pandas as pd 
import os
import json
import nltk
from sentence_transformers import SentenceTransformer
import pickle

train = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/train.csv')
p_train=train[['Id','cleaned_label','dataset_label']]

train_files_path='../input/coleridgeinitiative-show-us-the-data/train'
art_dic=pd.DataFrame(columns=['Id','sent','class'])
for i in range(p_train.shape[0]):
    #print('file number : ',i)
    filename=p_train['Id'][i]
    json_path = os.path.join(train_files_path, (filename+'.json'))
    contents = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            contents.append(data.get('text'))

    all_contents = ' '.join(contents)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    all_sent=tokenizer.tokenize(all_contents)
    #print ('number of sentences in article: ',len(all_sent))
    query = p_train['dataset_label'][i]
    for sent in all_sent:
            if sent.find(query) != -1 :
                new_row={'Id':p_train['Id'][i],'sent':str(sent),'class':query}
                art_dic=art_dic.append(new_row,ignore_index=True)
print('Done')
with open('artcl_sent_clss.pkl', "wb") as fOut:
    pickle.dump(art_dic, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    

