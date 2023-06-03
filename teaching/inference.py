import sys

sys.path.append("./")

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os, string, re

import hnswlib


class CFG:
    input_path = r'input_dir/'
    
    model_path = '/home/ftzheng/project/liuyilin/pre_ckpts/mdeberta-v3-base'
    step1_ckpt = 'output/title_InfoNce_step1.pth'
    step2_ckpt = 'output/title_n30_mdb_concat_step2.pth'
    sample = True  # 预测是 0.01 sample
    only_use_title = True
    use_parent_title = False
    max_input_length = 64
    batch_size = 256
    top_k = 50
    
    # model_path = 'mdeberta-v3-base'
    # step1_ckpt = ''
    # step2_ckpt = ''
    # sample = False
    # only_use_title = True
    # use_parent_title = False
    # max_input_length = 4
    # batch_size = 4
    # top_k = 30

print("="*30)
print("step1_ckpt: ", CFG.step1_ckpt)
print("step2_ckpt: ", CFG.step2_ckpt)
print("top-k: ", CFG.top_k)
print("="*30)


class ConcatTrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.max_input_length = max_input_length
        self.topic = df['topic_full_text'].values
        self.content = df['content_full_text'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
    
    def __len__(self):
        return len(self.topic)
    
    def __getitem__(self, item):
        topic = self.topic[item].replace('[SEP]', self.sep_token)
        content = self.content[item].replace('[SEP]', self.sep_token)
        concats = topic + " " + self.sep_token + " " + content
        
        label = 0
        
        inputs_concats = self.tokenizer(concats, truncation=True, max_length=self.max_input_length * 2,
                                        padding='max_length')
        
        return torch.as_tensor(inputs_concats['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs_concats['attention_mask'], dtype=torch.long), \
               torch.as_tensor(label, dtype=torch.float)


class Custom_Bert(nn.Module):
    """only used in eval - infer step"""
    
    def __init__(self, model_path):
        super().__init__()
        
        self.base = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
    
    def forward(self,
                input_ids,
                attention_mask=None):
        output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state
        attention_mask = torch.unsqueeze(attention_mask, 2)
        output *= attention_mask
        output = torch.sum(output, dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)
        output /= attention_mask
        return output


class TopicTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.max_input_length = max_input_length
        self.input = df['topic_full_text'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, item):
        input_text = self.input[item]
        output = self.tokenizer(input_text, truncation=True, max_length=self.max_input_length, padding='max_length')
        
        return torch.as_tensor(output['input_ids'], dtype=torch.long), \
               torch.as_tensor(output['attention_mask'], dtype=torch.long)


class ContentTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.max_input_length = max_input_length
        self.input = df['content_full_text'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, item):
        input_text = self.input[item]
        output = self.tokenizer(input_text, truncation=True, max_length=self.max_input_length, padding='max_length')
        
        return torch.as_tensor(output['input_ids'], dtype=torch.long), \
               torch.as_tensor(output['attention_mask'], dtype=torch.long)


def recall(targets, preds):
    return len([x for x in targets if x in preds]) / (len(targets) + 1e-16)


def f2_score(y_true, y_pred):
    y_true = [set(i.split()) for i in y_true]
    y_pred = [set(i.split()) for i in y_pred]
    
    tp, fp, fn = [], [], []
    for x in zip(y_true, y_pred):
        tp.append(np.array([len(x[0] & x[1])]))
        fp.append(np.array([len(x[1] - x[0])]))
        fn.append(np.array([len(x[0] - x[1])]))
    tp, fp, fn = np.array(tp), np.array(fp), np.array(fn)
    
    # precision = tp / (tp + fp)
    recs = [recall(t, p) for t, p in list(zip(y_true, y_pred))]
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4), np.nanmean(recs)


def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.2, 0.001):
        x_val['predictions'] = np.where(val_predictions > thres, 1, 0)
        x_val1 = x_val[x_val['predictions'] == 1]
        x_val1 = x_val1.groupby(['topic_id'])['content_id'].unique().reset_index()
        x_val1['content_id'] = x_val1['content_id'].apply(lambda x: ' '.join(x))
        x_val1.columns = ['topic_id', 'predictions']
        x_val0 = pd.Series(x_val['topic_id'].unique())
        x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
        x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
        x_val_r = pd.concat([x_val1, x_val0], axis=0, ignore_index=True)
        x_val_r = x_val_r.merge(correlations, how='left', on='topic_id')
        
        x_val_r = x_val_r.dropna().reset_index(drop=True)
        score, rec = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        # print(f"threshold:{thres}, score: {score:.3f}, recall: {rec:.3f}")
        
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold


class Concat_Bert(nn.Module):
    def __init__(self, model_path, represent_method="last_mean"):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.represent_method = represent_method
    
    def forward(self,
                input_ids,
                attention_mask=None,
                labels=None,
                output_logistics=False):
        
        # get concatenated sentence embedding
        output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        if self.represent_method == "last_mean":
            output = self.drop(output.last_hidden_state)
            # todo: get mean
            attention_mask = torch.unsqueeze(attention_mask, 2)  # batch_size, seq_len, 1
            # print(topic_output_mask)
            output *= attention_mask  # batch_size, seq_len, hidden_size
            output = torch.sum(output, dim=1)  # batch_size, hidden_size
            attention_mask = torch.sum(attention_mask, dim=1)
            output /= attention_mask
        elif self.represent_method == "cls":
            output = output[:, 0]
        
        # get predicted logits
        logits = self.linear(self.drop(output))
        
        # get loss and logits
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1))
        
        if output_logistics:
            return loss, logits
        else:
            return loss


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


def build_index(embeddings, ids):
    index = hnswlib.Index(space="cosine", dim=embeddings.shape[-1])
    index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=1000)
    index.set_ef(1000)
    index.set_num_threads(16)
    index.add_items(embeddings, ids)
    return index


def clean_text(text):
    for punctuation in list(string.punctuation): text = text.replace(punctuation, '')
    output = re.sub('\r+', ' ', text)
    output = re.sub('\n+', ' ', output)
    
    return output


def get_data(CFG, topic_df, content_df, sample=True):
    if not sample:
        topic_df = topic_df[topic_df['id'].isin(sub_df['topic_id'])]
    else:
        topic_df = topic_df.sample(frac=0.01).reset_index(drop=True)
    print(f'after filtering, the topic_df len is  {len(topic_df)}')
    
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
    
    return topic_df, content_df


def infer(model, dataloader, device, test_index=-1):
    """
    get all the representation of the dataset
    """
    res = []
    for i, batch in enumerate(tqdm(dataloader)):
        # for i, batch in enumerate(dataloader):
        if i == test_index: break
        input_ids, attention_mask = [i.to(device) for i in batch]
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            res.append(output.cpu().numpy())
    
    return np.vstack(res)


def get_step1_outputs(model, topic_loader, content_loader, content_df, topic_df, device, top_k=5):
    print(f"Each topics get top {top_k} candidates.")
    model.eval()
    with torch.no_grad():
        topic_result = infer(model, topic_loader, device)
        content_result = infer(model, content_loader, device)
        content_ids = [i for i in range(len(content_result))]
        content_index = build_index(content_result, content_ids)
        results = content_index.knn_query(topic_result, k=top_k,
                                          num_threads=-1)  # 返回的是 content_id; 索引是 topic_loader 的顺序
    
    pred = []
    content_uid = content_df['id']
    for result in tqdm(results[0], desc="aggregate predictions", mininterval=120):
        top_same = ' '.join(content_uid[result].to_list())  # 通过 content_id 得到 content_uid
        pred.append(top_same)
    
    topic_df['pred_content_id'] = pred
    # free memory
    return topic_df[['id', 'pred_content_id']]


if __name__ == '__main__':
    # TODO: golden data
    topic_df = pd.read_csv(os.path.join(CFG.input_path, 'topics.csv'))
    content_df = pd.read_csv(os.path.join(CFG.input_path, 'content.csv'))
    sub_df = pd.read_csv(os.path.join(CFG.input_path, 'sample_submission.csv'))
    
    if not CFG.sample:
        print("cut content_df for local test")
        content_df = content_df[:30]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_model = Custom_Bert(CFG.model_path)
    
    if CFG.step1_ckpt:
        print(f"\nstep1 model load from {CFG.step1_ckpt}\n")
        eval_model.load_state_dict(torch.load(CFG.step1_ckpt, map_location='cpu'), strict=False)
    eval_model.to(device)
    eval_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
    
    # TODO: get data
    topic_df, content_df = get_data(CFG, topic_df, content_df, sample=CFG.sample)
    
    topic_dataset = TopicTestDataset(topic_df, tokenizer, CFG.max_input_length)
    content_dataset = ContentTestDataset(content_df, tokenizer, CFG.max_input_length)
    topic_eval_loader = DataLoader(topic_dataset,
                                   batch_size=CFG.batch_size * 2,
                                   shuffle=False,
                                   num_workers=4 if torch.cuda.is_available() else 1,
                                   pin_memory=True, drop_last=False)
    content_eval_loader = DataLoader(content_dataset,
                                     batch_size=CFG.batch_size * 2,
                                     shuffle=False,
                                     num_workers=4 if torch.cuda.is_available() else 1,
                                     pin_memory=True, drop_last=False)
    
    # TODO: get and save results
    step1_result = get_step1_outputs(eval_model, topic_eval_loader, content_eval_loader,
                                     content_df, topic_df, device, top_k=CFG.top_k)
    
    #  TODO: Step2
    eval_model = Concat_Bert(CFG.model_path, represent_method="last_mean").to(device)
    
    if CFG.step2_ckpt:
        print(f"\nstep2 model load from {CFG.step2_ckpt}\n")
        eval_model.load_state_dict(torch.load(CFG.step2_ckpt, map_location='cpu'), strict=False)
    eval_model.to(device)
    eval_model.eval()
    
    step1_result['pred_content_id'] = step1_result['pred_content_id'].apply(lambda x: list(set(x.split(" "))))
    step1_df = step1_result.explode("pred_content_id").reset_index(drop=True)
    
    content_df = content_df.rename(columns={'language': 'content_language'})
    topic_df = topic_df.rename(columns={'language': 'topic_language'})
    step1_df = step1_df.rename(columns={'id': 'topic_id', 'pred_content_id': 'content_id'})
    step1_df = step1_df.merge(topic_df[["id", "topic_full_text", "topic_language"]], left_on="topic_id", right_on="id")
    step1_df = step1_df.merge(content_df[["id", "content_full_text", "content_language"]], left_on="content_id",
                              right_on="id")
    step1_df = step1_df[step1_df["topic_language"] == step1_df["content_language"]]
    step1_df = step1_df[['topic_id', 'content_id', 'topic_full_text', 'content_full_text']]
    
    tr_dataset = ConcatTrainDataset(step1_df, tokenizer, CFG.max_input_length)
    fine_loader = DataLoader(tr_dataset,
                             batch_size=CFG.batch_size * 2,
                             shuffle=False,
                             num_workers=1,
                             # pin_memory=True,
                             drop_last=False)
    
    all_logistics = []
    for batch in tqdm(fine_loader):
        batch = [i.to(device) for i in batch]
        
        with torch.no_grad():
            concats_input_ids, concats_attention_mask, _ = batch
            _, logistics = eval_model(concats_input_ids, concats_attention_mask,
                                      None,
                                      output_logistics=True)
            all_logistics.append(logistics)
    
    all_logistics = torch.vstack(all_logistics)
    all_logistics = all_logistics.detach().cpu().squeeze()
    all_logistics = torch.sigmoid(all_logistics)
    
    corr_df = pd.read_csv(os.path.join(CFG.input_path, 'correlations.csv'))
    best_score, best_threshold = get_best_threshold(step1_df, all_logistics, corr_df)
    print(f"The Best Threshold is {best_threshold} with Score {best_score}")
