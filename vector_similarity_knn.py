import pandas as pd
import numpy as np
import json
import string
import nltk
from sentence_transformers import SentenceTransformer,util
import math
import pickle
#import keras
import hnswlib
import os


train=pd.DataFrame(columns=['Id','sent','sent_vec','class','sim_prc'])
with open('artcl_sent_clss.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    train = pd.DataFrame(stored_data)
    corpus_sentences = stored_data['sent']
    corpus_class=stored_data['class']


q_model = SentenceTransformer('quora-distilbert-multilingual')
max_corpus_size = 100000
embedding_cache_path = 'embed_sent.pkl'

if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    corpus_sentences = list(corpus_sentences)
    corpus_embeddings = q_model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open('embed_sent.pkl', "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings,'class':corpus_class}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']
        corpus_class=cache_data['class']


embedding_size = 768    #Size of embeddings
top_k_hits = 10 

index_path = ".\\q_hnswlib.index"
#We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
index = hnswlib.Index(space = 'cosine', dim = embedding_size)

if os.path.exists(index_path):
    print("Loading index...")
    index.load_index(index_path)
else:
    ### Create the HNSWLIB index
    print("Start creating HNSWLIB index")
    index.init_index(max_elements = len(corpus_embeddings), ef_construction = 400, M = 64)

    # Then we train the index to find a suitable clustering
    index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))

    print("Saving index to:", index_path)
    index.save_index(index_path)

# Controlling the recall by setting ef:
index.set_ef(50)  # ef should always be > top_k_hits

test=pd.read_csv('processedTest_v2.csv',header=0)
p_test=test[['Id','text','PredictionString']]

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
        inp_question = sent
        question_embedding = q_model.encode(inp_question)
        #We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = index.knn_query(question_embedding, k=top_k_hits)
        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        sent_class=pd.DataFrame(columns=['score','corpus_sent','corpus_class'])
        for hit in hits[0:top_k_hits]:
            new_row={'score':hit['score'],'corpus_sent':corpus_sentences[hit['corpus_id']],'corpus_class':corpus_class[hit['corpus_id']]}
            sent_class=sent_class.append(new_row,ignore_index=True)

        sent_class.reset_index(inplace=True)
        sent_class=sent_class[['index','score','corpus_class']]
        df_class=pd.DataFrame(sent_class.groupby('corpus_class').agg({'score':"mean",'index':"min"})).reset_index()
        df_class['last_score']= df_class['score'] *(10 - df_class['index'])
        df_class=df_class.sort_values(by=['last_score'],ascending=False).reset_index()
        new_row={'Id':p_test['Id'][i],'class':df_class['corpus_class'][0],'score':df_class['last_score'][0]}
        #print(new_row)
        art_dataset=art_dataset.append(new_row,ignore_index=True)

    art_dataset=art_dataset.sort_values(by='score', ascending=False)
    top_art_hits=30
    art_dataset=art_dataset[0:top_art_hits].reset_index(drop=True)
    art_dataset.reset_index(inplace=True)
    df_art=pd.DataFrame(art_dataset.groupby('class').agg({'score':"mean",'index':"min"})).reset_index()
    df_art['last_score']= df_art['score'] *(30 - df_art['index'])
    df_art=df_art.sort_values(by=['last_score'],ascending=False).reset_index()
    new_art_row={'Id':p_test['Id'][i],'class':df_art['class'][0],'score':df_art['last_score'][0]}
    print(new_art_row)
    whl_dataset=whl_dataset.append(new_art_row,ignore_index=True)


whl_dataset.to_csv('vector_similarity_knn.csv')  
print('Done done')