import numpy as np
import pandas as pd
from rank_bm25 import BM25L
from tqdm import tqdm
from transformers import AutoTokenizer


def get_bm25_negative(sentence, candidates_sentence, thres=2):
    tokenized_corpus = [tokenizer.tokenize(i) for i in candidates_sentence]
    bm25 = BM25L(tokenized_corpus)
    tokenized_query = tokenizer.tokenize(sentence)
    doc_scores = bm25.get_scores(tokenized_query)
    index = np.argpartition(doc_scores, -2)[-2:]
    return np.array(candidates_sentence)[index].tolist()


input_dic = r'/home/ftzheng/project/liuyilin/kaggle220131/input_dir/'
topic_df = pd.read_csv(input_dic + r'topics.csv')
content_df = pd.read_csv(input_dic + r'content.csv')
corr_df = pd.read_csv(input_dic + r'correlations.csv')
topic_df = topic_df.rename(columns={'id': 'topic_id'}).merge(corr_df)

tokenizer = AutoTokenizer.from_pretrained('/home/ftzheng/project/liuyilin/pre_ckpts/mdeberta-v3-base')
corr_df['content_ids'] = corr_df['content_ids'].apply(lambda x: x.split())
corr_df = corr_df.explode('content_ids').reset_index(drop=True)
topic_df = topic_df.fillna('')
topic_df['topic_full_text'] = topic_df['title'] + ' [SEP] ' + topic_df['description']
topic_df = topic_df[['topic_id', 'topic_full_text', 'language', 'title']]
df = corr_df.merge(topic_df, left_on='topic_id', right_on='topic_id', how='left')
df = df[['topic_id', 'content_ids', 'topic_full_text', 'language', 'title']]
df = df.rename(columns={'language': 'topic_language', 'title': 'topic_title'})
content_df = content_df.fillna('')
content_df['content_full_text'] = content_df['title'] + ' [SEP] ' + content_df['description'] + ' [SEP] ' + content_df[
    'text']
content_df = content_df[['id', 'content_full_text', 'language', 'title']]
df = df.merge(content_df, left_on='content_ids', right_on='id', how='left')
df = df.rename(columns={'language': 'content_language', 'title': 'content_title'})
df['label'] = 1
neg_df = []
for topic_id in tqdm(df['topic_id'].unique()[:100]):
    sub_df = df[df['topic_id'] == topic_id]
    topic_language = sub_df['topic_language'].unique()[0]
    topic_full_text = sub_df['topic_full_text'].unique()[0]
    ## bm25
    bm25 = []
    
    querys = sub_df['content_full_text'].to_list()
    for i in querys:
        candidates = df[df['content_language'] == topic_language].sample(n=30)
        candidates_sentences = candidates['content_full_text'].to_list()
        bm25_negative = pd.DataFrame({'topic_full_text': [topic_full_text],
                                      'content_full_text': ['']
                                      })
        results = get_bm25_negative(i, candidates_sentences)
        bm25_negative['content_full_text'] = '\u00001'.join(results)
        bm25_negative['topic_id'] = topic_id
        bm25_negative['label'] = 0
        bm25_negative['content_full_text'] = bm25_negative['content_full_text'].apply(lambda x: x.split('\u00001'))
        bm25_negative = bm25_negative.explode('content_full_text').reset_index(drop=True)
        bm25.append(bm25_negative)
    
    bm25 = pd.concat(bm25)
    bm25 = bm25[-(bm25['content_full_text'].isin(sub_df['content_full_text'].to_list()))]
    
    neg_df.append(bm25)

neg_df = pd.concat(neg_df)
neg_df = neg_df.drop_duplicates()
neg_df.to_csv(input_dic + r'bm25_negative.csv', index=False)
