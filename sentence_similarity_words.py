import numpy as np
import pandas as pd 
import os
import json
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

train=pd.DataFrame(columns=['Id','sent','class'])
with open('artcl_sent_clss.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    train = pd.DataFrame(stored_data)
    corpus_sentences = stored_data['sent']
    corpus_class=stored_data['class']

'''
art_agg=pd.DataFrame(stored_data.groupby('Id')['class'].nunique()).reset_index()
art_agg_cnt=pd.DataFrame(art_agg.groupby(['class']).count()).reset_index()
art_agg_cnt['percent'] = (art_agg_cnt['Id'].cumsum() / art_agg_cnt['Id'].sum()) * 100
class_per_Id=art_agg_cnt[art_agg_cnt['percent']> 99]['class'].min() 
##### more than 99% of articles have 3 datasets or less
sent_agg=pd.DataFrame(stored_data.groupby('sent')['class'].nunique()).reset_index()
sent_agg_cnt=pd.DataFrame(sent_agg.groupby(['class']).count()).reset_index()
sent_agg_cnt['percent'] = (sent_agg_cnt['sent'].cumsum() / sent_agg_cnt['sent'].sum()) * 100
class_per_sent=sent_agg_cnt[sent_agg_cnt['percent']>99]['class'].min() 
##### more than 99% of sentences have 2 datasets or less
'''

def sent_class(sent_id,x):
    sent_sim=pd.DataFrame(columns=['sent_x','sent_y','class','sim_prc'])
    for i in stored_data['class'].unique():
        y=stored_data[stored_data['class']== i]['sent'].values
        if len(y) != 0:
            x = ''.join([k for k in x if k not in string.punctuation]).lower()
            y=''.join([k for k in str(y[0]) if k not in string.punctuation]).lower()
            X_list = word_tokenize(x)
            Y_list = word_tokenize(y)
            # sw contains the list of stopwords
            sw = stopwords.words('english') 
            lx =[];ly =[]
            # remove stop words from the string
            X_set = {w for w in X_list if not w in sw} 
            Y_set = {w for w in Y_list if not w in sw}
            # form a set containing keywords of both strings 
            rvector = X_set.union(Y_set) 
            for w in rvector:
                if w in X_set: lx.append(1) # create a vector
                else: lx.append(0)
                if w in Y_set: ly.append(1)
                else: ly.append(0)
            c = 0
            # cosine formula 
            for j in range(len(rvector)):
                    c+= lx[j]*ly[j]
            cosine = c / float((sum(lx)*sum(ly))**0.5)
            new_row={'sent_x':x,'sent_y':y,'class':i,'sim_prc':cosine}
            sent_sim=sent_sim.append(new_row,ignore_index=True)

    top_k_hits=10
    sent_sim=sent_sim.sort_values(by=['sim_prc'],ascending=False).reset_index()
    sent_sim=sent_sim[:top_k_hits]
    sent_sim['hscore']=(10 - sent_sim['index'])*sent_sim['sim_prc']
    sent_sim['count']=1
    df_sent=pd.DataFrame(sent_sim.groupby('class').agg({'hscore':"mean",'count':"sum"})).reset_index()
    df_sent['last_score']=df_sent['hscore']*df_sent['count']
    df_sent=df_sent.sort_values(by=['last_score'],ascending=False).reset_index()
    return({'Id':sent_id,'class':df_sent['class'][0],'score':df_sent['last_score'][0]},
    {'Id':sent_id,'class':df_sent['class'][1],'score':df_sent['last_score'][1]})


p_test=pd.read_csv('sample_submission.csv',header=0)
whl_dataset=pd.DataFrame(columns=['Id','class','score'])
test_files_path='D:\\Self_Study\\Kaggle_Coleridge Initiative\\test'
for i in range(p_test.shape[0]):
    print('file number : ',i)
    filename=p_test['Id'][i]
    json_path = os.path.join(test_files_path, (filename+'.json'))
    contents = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            contents.append(data.get('text'))

    all_contents = ' '.join(contents)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    all_sent=tokenizer.tokenize(all_contents)
    print ('number of sentences in article: ',len(all_sent))
    art_dataset=pd.DataFrame(columns=['Id','class','score'])
    for sent in all_sent:
        new_row=pd.DataFrame(sent_class(p_test['Id'][i],sent))
        art_dataset=art_dataset.append(new_row,ignore_index=True)

    art_dataset=art_dataset.sort_values(by='score', ascending=False)
    top_art_hits=30
    art_dataset=art_dataset[0:top_art_hits].reset_index(drop=True)
    art_dataset.reset_index(inplace=True)
    art_dataset['hscore']=(30-art_dataset['index'])*art_dataset['score']
    art_dataset['count']=1
    df_art=pd.DataFrame(art_dataset.groupby('class').agg({'hscore':"mean",'count':"sum"})).reset_index()
    df_art['last_score']=df_art['hscore']*df_art['count']
    df_art=df_art.sort_values(by=['last_score'],ascending=False).reset_index()
    print(df_art)
    print('number 1:',df_art['class'][1])
    pred=df_art['class'][0]+ ('|'+df_art['class'][1] if df_art['class'][1] else '')+('|'+df_art['class'][2] if df_art['class'][2] else '')
    new_art_row={'Id':p_test['Id'][i],'PredictionString':pred}
    print(new_art_row)
    whl_dataset=whl_dataset.append(new_art_row,ignore_index=True)


whl_dataset.to_csv('sent_similarity_words.csv')  
print('Done Done')
