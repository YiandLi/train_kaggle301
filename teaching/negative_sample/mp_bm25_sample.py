# !pip install multiprocesspandas
# !pip install rank_bm25
import os
import re
import string
import sys
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25L
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append("./")
from utils.basic_utils import generate_parents_nodes_title


def clean_text(text):
    for punctuation in list(string.punctuation): text = text.replace(punctuation, '')
    output = re.sub('\r+', ' ', text)
    output = re.sub('\n+', ' ', output)
    
    return output


def get_logger():
    import logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


LOGGER = get_logger()


# for single sentence
def get_bm25_negative(sentence, candidates_sentence, tokenizer, thres=2):
    """
        para: candidates : sentenc list
        para: thres, is fixed , not used here

        using `tokenizer.tokenize(i)` as token ids
    """
    candidates_num = len(candidates_sentence)
    sample_num = min(candidates_num, thres)
    
    tokenized_corpus = [tokenizer.tokenize(i) for i in candidates_sentence]
    bm25 = BM25L(tokenized_corpus)
    tokenized_query = tokenizer.tokenize(sentence)
    doc_scores = bm25.get_scores(tokenized_query)
    index = np.argpartition(doc_scores, -sample_num)[-sample_num:]
    return np.array(candidates_sentence)[index].tolist()


# add cut_num, cut_index for multiprocess
def get_bm25_negtives(df, cut_num, cut_index, candidates_num, tokenizer, save_path, mode):
    """
    :param df:
    :param cut_num: 进程数量
    :param cut_index:  进程索引
    :param candidates_num:  根据同语言 粗采样 的个数，线性决定 bm25 的时间 ( n )
    :return: list of pd; each correspond to one unique topic_id
    """
    total_id_num = len(df['topic_id'].unique())
    start = cut_index * (total_id_num // cut_num)
    end = start + total_id_num // cut_num
    end = total_id_num if cut_index == (cut_num - 1) else end
    neg_df = []
    
    for topic_id in tqdm(df['topic_id'].unique()[start:end], desc=f"the {cut_index}th process: "):
        
        sub_df = df[df['topic_id'] == topic_id]
        topic_language = sub_df['topic_language'].unique()[0]
        topic_full_text = sub_df['topic_full_text'].unique()[0]
        ## bm25
        bm25 = []
        querys = sub_df['content_full_text'].to_list()  # bm25's query
        
        for i in querys:
            same_language_df = df[df['content_language'] == topic_language]
            # firstly roughly (of same language) sample candidates_num candidates
            candidates = same_language_df.sample(n=min(candidates_num, len(same_language_df)))
            
            candidates_sentences = candidates['content_full_text'].to_list()
            bm25_negative = pd.DataFrame({'topic_full_text': [topic_full_text],
                                          'content_full_text': ['']
                                          })
            results = get_bm25_negative(i, candidates_sentences, tokenizer)
            bm25_negative['content_full_text'] = '\u00001'.join(results)  # 空格
            bm25_negative['topic_id'] = topic_id
            bm25_negative['label'] = 0
            bm25_negative['content_full_text'] = bm25_negative['content_full_text'].apply(
                lambda x: x.split('\u00001'))
            bm25_negative = bm25_negative.explode('content_full_text').reset_index(drop=True)
            bm25.append(bm25_negative)
        
        bm25 = pd.concat(bm25)
        
        # filter positive
        bm25 = bm25[-(
            bm25['content_full_text'].isin(sub_df['content_full_text'].to_list())
        )]
        
        neg_df.append(bm25)
    
    neg_df = pd.concat(neg_df)
    neg_df = neg_df.drop_duplicates()
    
    if not os.path.exists(os.path.join(save_path, "bm25_neg_sample")):
        os.mkdir(os.path.join(save_path, "bm25_neg_sample"))
    
    output_path = os.path.join(save_path, "bm25_neg_sample", f"{mode}_neg_sample_{cut_index}.csv")
    neg_df.to_csv(output_path, index=False)
    
    LOGGER.info(f"\nthe {cut_index}th process done get {len(neg_df)} instances, saved in {output_path}\n\n")
    
    # return neg_df  # list of dataframe


if __name__ == '__main__':
    mode = "parent_title"  # {'title', 'all', 'parent_title}
    
    # tokenizer = AutoTokenizer.from_pretrained('mdeberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained('/home/ftzheng/project/liuyilin/pre_ckpts/mdeberta-v3-base')
    
    start = time.time()
    input_path = r'input_dir/'
    topic_df = pd.read_csv(input_path + r'topics.csv')
    content_df = pd.read_csv(input_path + r'content.csv')
    corr_df = pd.read_csv(input_path + r'correlations.csv')
    
    if mode == "parent_title":
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df = generate_parents_nodes_title(topic_df)
        topic_df['title'] = topic_df['topic_tree']
    
    topic_df = topic_df.rename(columns={'id': 'topic_id'}).merge(corr_df)
    corr_df['content_ids'] = corr_df['content_ids'].apply(lambda x: x.split())
    corr_df = corr_df.explode('content_ids').reset_index(drop=True)
    
    topic_df = topic_df.fillna('')
    content_df = content_df.fillna('')
    
    if mode == "all":
        topic_df['topic_full_text'] = topic_df['title'] + ' [SEP] ' + topic_df['description']
        content_df['content_full_text'] = content_df['title'] + ' [SEP] ' + content_df['description'] + ' [SEP] ' + \
                                          content_df['text']
    elif mode == "title" or mode == 'parent_title':
        topic_df['topic_full_text'] = topic_df['title']
        content_df['content_full_text'] = content_df['title']
    else:
        print("mode not find, please check")
        exit()
    
    topic_df = topic_df[['topic_id', 'topic_full_text', 'language', 'title']]
    df = corr_df.merge(topic_df, left_on='topic_id', right_on='topic_id', how='left')
    df = df[['topic_id', 'content_ids', 'topic_full_text', 'language', 'title']]
    df = df.rename(columns={'language': 'topic_language', 'title': 'topic_title'})
    
    topic_df['topic_full_text'] = topic_df['topic_full_text'].apply(lambda x: clean_text(x))
    content_df['content_full_text'] = content_df['content_full_text'].apply(lambda x: clean_text(x))
    
    content_df = content_df[['id', 'content_full_text', 'language', 'title']]
    df = df.merge(content_df, left_on='content_ids', right_on='id', how='left')
    df = df.rename(columns={'language': 'content_language', 'title': 'content_title'})
    df['label'] = 1
    
    # print(df.head())
    
    # for test
    # df = df[:500]
    
    # TODO: multi_process
    from multiprocessing import Pool
    
    # 多进程数量
    process_num = 5
    rough_candidates_num = 500  # 同语言粗粒度 sample 的个数； 细粒度的默认为 2 `thres`
    p = Pool(process_num)
    neg_dfs = []
    for i in range(process_num):
        p.apply_async(get_bm25_negtives, args=(df, process_num, i, rough_candidates_num, tokenizer, input_path, mode))
    p.close()
    p.join()
    
    #     neg_dfs.append(result)
    #
    # neg_dfs = [i.get() for i in neg_dfs]
    #
    # # save
    # output_path = os.path.join(input_path, f"mp_bm25_output_{rough_candidates_num}_2.csv")
    # neg_df = pd.concat(neg_dfs)
    # neg_df = neg_df.drop_duplicates()
    # neg_df.to_csv(output_path, index=False)
    # LOGGER.info(f"totally {len(neg_df)} instances, saved in {output_path}")
