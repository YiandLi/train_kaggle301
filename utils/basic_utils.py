import os
import random
import re
import string

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

from utils.datasets import TopicTestDataset, ContentTestDataset


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger():
    import logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def clean_text(text):
    for punctuation in list(string.punctuation): text = text.replace(punctuation, '')
    output = re.sub('\r+', ' ', text)
    output = re.sub('\n+', ' ', output)
    
    return output


def get_data(CFG, step: str):
    """
    维护三个 df
        folds -> topic_df['stratify'] 做采样 ： 仅包含 ['topic_id', 'fold']
        df -> 整合 topic 和 content 的输入文本
        neg_df -> 直接拿内部的 'topic_full_text', 'content_full_text'
    """
    # TODO: golden data
    topic_df = pd.read_csv(os.path.join(CFG.input_path, 'topics.csv'))
    content_df = pd.read_csv(os.path.join(CFG.input_path, 'content.csv'))
    corr_df = pd.read_csv(os.path.join(CFG.input_path, 'correlations.csv'))
    
    # Set fold
    topic_df['stratify'] = topic_df['category'] + topic_df['language'] + \
                           topic_df['description'].apply(lambda x: str(isinstance(x, str))) + \
                           topic_df['has_content'].apply(str)
    
    # TODO: 给 topic 分组 这里不用 group（"channel"）信息了 --》 fold_df
    kf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=42)
    folds = list(
        kf.split(topic_df,
                 y=topic_df["stratify"],
                 ))
    
    topic_df['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        topic_df.loc[val_idx, "fold"] = fold
    
    # 整合 non-source 的 fold id
    fold_df = topic_df[['id', 'fold']].reset_index(drop=True)
    # 将 source 的 fold id 设定为 -1
    fold_df = fold_df.rename(columns={'id': 'topic_id'})
    fold_df['fold'] = fold_df['fold'].astype(int)  # 整合了所有的 topic id 和 fold id
    
    # TODO：整合输入文本
    corr_df['content_ids'] = corr_df['content_ids'].apply(lambda x: x.split())
    corr_df = corr_df.explode('content_ids').reset_index(drop=True)
    
    if CFG.use_parent_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df = generate_parents_nodes_title(topic_df)
        topic_df['title'] = topic_df['topic_tree']
        topic_df = topic_df.drop(columns=['parent', 'topic_tree'])
    
    if CFG.only_use_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df['topic_full_text'] = topic_df['title']
        content_df = content_df.dropna(subset=['title']).reset_index(drop=True)
        content_df['content_full_text'] = content_df['title']
    else:
        topic_df = topic_df.fillna('')
        topic_df['topic_full_text'] = topic_df['title'] + ' [SEP] ' + topic_df['description']
        content_df = content_df.fillna('')
        content_df['content_full_text'] = content_df['title'] + ' [SEP] ' + content_df['description'] + ' [SEP] ' + \
                                          content_df['text']
    
    topic_df['topic_full_text'] = topic_df['topic_full_text'].apply(lambda x: clean_text(x))
    content_df['content_full_text'] = content_df['content_full_text'].apply(lambda x: clean_text(x))
    
    topic_df = topic_df[['id', 'title', 'topic_full_text', 'language']]
    content_df = content_df[['id', 'title', 'content_full_text', 'language']]
    
    # TODO：整合 topic 和 content
    df = corr_df.merge(topic_df, left_on='topic_id', right_on='id', how='left')
    df = df[['topic_id', 'content_ids', 'topic_full_text']]
    df = df.merge(content_df, left_on='content_ids', right_on='id', how='left')
    df['label'] = 1
    df = df.merge(fold_df, on='topic_id', how='left')
    df = df[['topic_full_text', 'content_full_text', 'topic_id', 'label', 'fold']]
    df = df.dropna().reset_index(drop=True)  # 其中的 topic id 都有非空的 content 和他对应
    
    if step == "step2":
        if CFG.use_parent_title:
            neg_child_dir_path = "parent_title_negs"
        elif CFG.only_use_title:
            neg_child_dir_path = "title_negs"
        else:
            neg_child_dir_path = "all_negs"
        
        neg_dir = os.path.join(CFG.negative_dir_path, neg_child_dir_path)
        negtive_datas = os.listdir(neg_dir)
        
        # TODO: negative_data
        for neg_path in negtive_datas:
            print(f"Read negatives from {os.path.join(neg_dir, neg_path)}")
            neg_df = pd.read_csv(os.path.join(neg_dir, neg_path))
            
            neg_df = neg_df.merge(fold_df, on='topic_id', how='left')
            neg_df = neg_df.dropna().reset_index(drop=True)
            
            neg_df = neg_df[['topic_full_text', 'content_full_text', 'topic_id', 'label', 'fold']]
            print(f"This negative samples fold id count is: {neg_df.groupby('fold').count()['topic_id'].to_dict()} ")
            df = pd.concat([df, neg_df])
    
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle df
    print(f"\nThe total (pos+neg) fold id count is: {df.groupby('fold').count()['topic_id'].to_dict()}")
    print(f"The total instances label count is: {df.groupby('label').count()['topic_id'].to_dict()}, "
          f"0 for negative pairs ; 1 for positive pairs \n")
    
    return df


def get_step1_eval_data(eval_df, tokenizer, CFG):
    # get_data
    content_df = pd.read_csv(CFG.input_path + 'content.csv')
    topic_df = pd.read_csv(CFG.input_path + 'topics.csv')
    corr_df_init = pd.read_csv(CFG.input_path + 'correlations.csv')
    
    if CFG.use_parent_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df = generate_parents_nodes_title(topic_df)
        topic_df['title'] = topic_df['topic_tree']
        topic_df = topic_df.drop(columns=['parent', 'topic_tree'])
    
    # process df
    val_topic_id = eval_df['topic_id'].unique().tolist()  # eval中的topic id
    topic_df = topic_df[topic_df['id'].isin(val_topic_id)]  # 根据 选出所有在 val_topic_id 中的 topic_df
    
    if CFG.only_use_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df['topic_full_text'] = topic_df['title']
        content_df = content_df.dropna(subset=['title']).reset_index(drop=True)
        content_df['content_full_text'] = content_df['title']
    else:
        topic_df = topic_df.fillna('')
        topic_df['topic_full_text'] = topic_df['title'] + ' [SEP] ' + topic_df['description']
        content_df = content_df.fillna('')
        content_df['content_full_text'] = content_df['title'] + ' [SEP] ' + content_df['description'] + ' [SEP] ' + \
                                          content_df['text']
    
    topic_df['topic_full_text'] = topic_df['topic_full_text'].apply(lambda x: clean_text(x))
    content_df['content_full_text'] = content_df['content_full_text'].apply(lambda x: clean_text(x))
    
    topic_dataset = TopicTestDataset(topic_df, tokenizer, CFG.max_input_length)
    content_dataset = ContentTestDataset(content_df, tokenizer, CFG.max_input_length)
    topic_loader = DataLoader(topic_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=4 if torch.cuda.is_available() else 1,
                              pin_memory=True, drop_last=False)
    content_loader = DataLoader(content_dataset,
                                batch_size=CFG.batch_size * 2,
                                shuffle=False,
                                num_workers=4 if torch.cuda.is_available() else 1,
                                pin_memory=True, drop_last=False)
    
    assert len(content_dataset) == len(content_df), \
        f"The length of content_dataset [{len(content_dataset)}] " \
        f"could not match the length of content_df [{len(content_df)}]"
    
    # label
    corr_df_init = corr_df_init[corr_df_init['topic_id'].isin(val_topic_id)]  # 筛选出 val_topics 的正确答案
    gts = topic_df.merge(corr_df_init, how='left', left_on='id', right_on='topic_id')[
        'content_ids'].to_list()  # 正确答案合并到 topic_df（ 和topic_loader的顺序是相同的 ）
    
    return topic_loader, content_loader, topic_df, content_df, gts


def get_step2_pred_df(CFG, use_golden=True):
    # TODO: get_data
    # topic-id 'id', 'pred_content_id', 'true_content_id'
    # step1_df = pd.read_csv(CFG.step2_inputs, error_bad_lines=False)[:10]
    print("=" * 30)
    print(f"read data from {CFG.step2_inputs} ")
    step1_df = pd.read_csv(CFG.step2_inputs)
    content_df = pd.read_csv(CFG.input_path + 'content.csv')
    topic_df = pd.read_csv(CFG.input_path + 'topics.csv')
    
    step1_df = step1_df.dropna()
    
    # 正样本全使用
    if use_golden:
        step1_df['pred_content_id'] = step1_df['pred_content_id'] + step1_df['true_content_id']
        print("Using all golden contents as the positive instances")
    
    step1_df['pred_content_id'] = step1_df['pred_content_id'].apply(lambda x: list(set(x.split(" "))))
    step1_df['true_content_id'] = step1_df['true_content_id'].apply(lambda x: list(set(x.split(" "))))
    step1_df = step1_df.explode("pred_content_id").reset_index(drop=True)
    step1_df['label'] = step1_df.apply(lambda x: 1 if str(x.pred_content_id) in x.true_content_id else 0, axis=1)
    
    if CFG.use_parent_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df = generate_parents_nodes_title(topic_df)
        topic_df['title'] = topic_df['topic_tree']
        topic_df = topic_df.drop(columns=['parent', 'topic_tree'])
    
    if CFG.only_use_title:
        topic_df = topic_df.dropna(subset=['title']).reset_index(drop=True)
        topic_df['topic_full_text'] = topic_df['title']
        content_df = content_df.dropna(subset=['title']).reset_index(drop=True)
        content_df['content_full_text'] = content_df['title']
    else:
        topic_df = topic_df.fillna('')
        topic_df['topic_full_text'] = topic_df['title'] + ' [SEP] ' + topic_df['description']
        content_df = content_df.fillna('')
        content_df['content_full_text'] = content_df['title'] + ' [SEP] ' + content_df['description'] + ' [SEP] ' + \
                                          content_df['text']
    
    topic_df['topic_full_text'] = topic_df['topic_full_text'].apply(lambda x: clean_text(x))
    content_df['content_full_text'] = content_df['content_full_text'].apply(lambda x: clean_text(x))
    
    content_df = content_df.rename(columns={'language': 'content_language'})
    topic_df = topic_df.rename(columns={'language': 'topic_language'})
    step1_df = step1_df.rename(columns={'id': 'topic_id', 'pred_content_id': 'content_id'})
    step1_df = step1_df.merge(topic_df[["id", "topic_full_text", "topic_language"]], left_on="topic_id", right_on="id")
    step1_df = step1_df.merge(content_df[["id", "content_full_text", "content_language"]], left_on="content_id",
                              right_on="id")
    print(f"Step_1 return {len(step1_df)} instances. ")
    
    step1_df = step1_df[step1_df["topic_language"] == step1_df["content_language"]]
    print(f"After language filtering, still have {len(step1_df)} instances. ")
    
    step1_df = step1_df[['topic_id', 'content_id', 'topic_full_text', 'content_full_text', 'label']]
    step1_df = step1_df.dropna().reset_index(drop=True)
    print(f"After dropna() filtering, still have {len(step1_df)} instances. ")
    
    # TODO: Set fold id
    kf = GroupKFold(n_splits=8)  # StratifiedKFold （, shuffle=True, random_state=42）
    folds = list(
        kf.split(step1_df,
                 # y=step1_df["topic_id"],
                 groups=step1_df["topic_id"].values,
                 ))
    
    step1_df['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        step1_df.loc[val_idx, "fold"] = fold
    
    return step1_df


def generate_parents_nodes_title(topics):
    global nodes
    df = pd.DataFrame()
    for channel in topics['channel'].unique():
        channel_df = topics[(topics['channel'] == channel)].reset_index(drop=True)
        for level in sorted(channel_df.level.unique()):
            # For level 0, it first creates a topic tree column which is the title of that topic.
            if level == 0:
                topic_tree = channel_df[channel_df['level'] == level]['title'].astype(str)
                
                topic_tree_df = pd.DataFrame(
                    [channel_df[channel_df['level'] == level][['id']].values.squeeze(0), topic_tree.values]).T
                topic_tree_df.columns = ['child_id', 'topic_tree']
                channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(
                    ['child_id'], axis=1)
            
            # Once the topic tree column has been created, the parent node and child node is merged on parent_id = child_id
            topic_df_parent = channel_df[channel_df['level'] == level][['id', 'title', 'parent', 'topic_tree']]
            topic_df_parent.columns = 'parent_' + topic_df_parent.columns
            
            topic_df_child = channel_df[channel_df['level'] == level + 1][['id', 'title', 'parent', 'topic_tree']]
            topic_df_child.columns = 'child_' + topic_df_child.columns
            
            topic_df_merged = topic_df_parent.merge(topic_df_child, left_on='parent_id', right_on='child_parent')[
                ['child_id', 'parent_id', 'parent_title', 'child_title', 'parent_topic_tree']]
            
            # Topic tree is parent topic tree + title of the current child on that level
            topic_tree = topic_df_merged['parent_topic_tree'].astype(str) + ' [SEP] ' + topic_df_merged[
                'child_title'].astype(str)
            
            topic_tree_df = pd.DataFrame([topic_df_merged['child_id'].values, topic_tree.values]).T
            topic_tree_df.columns = ['child_id', 'topic_tree']
            
            channel_df = channel_df.merge(topic_tree_df, left_on='id', right_on='child_id', how='left').drop(
                ['child_id'], axis=1)
            if 'topic_tree_y' in list(channel_df.columns):
                channel_df['topic_tree'] = channel_df['topic_tree_x'].combine_first(channel_df['topic_tree_y'])
                channel_df = channel_df.drop(['topic_tree_x', 'topic_tree_y'], axis=1)
        
        df = pd.concat([df, channel_df])
    
    return df


def get_model_name(CFG, step):
    # get model name
    model_name = ""
    if CFG.use_parent_title:
        model_name += "parent_"
    if CFG.only_use_title:
        model_name += "title_"
    if step == "step2":
        model_name += f"n{CFG.top_k}_"
        # PLM model
        if "mdeberta" in CFG.model_path:
            model_name += "mdb_"
        elif "roberta" in CFG.model_path:
            model_name += "rbt_"
    
    model_name += f"{CFG.model_type}_{step}.pth"
    return model_name
