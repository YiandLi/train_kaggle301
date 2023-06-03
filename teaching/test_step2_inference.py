import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.datasets import TrainDataset
from utils.models import Custom_Bert_Simple

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_fine_inputs(rough_outs):
    # rough_outs = pd.read_csv("teaching/submission.csv")
    topics = pd.read_csv("topics.csv")
    contents = pd.read_csv("contents.csv")
    
    rough_outs["content_ids"] = rough_outs["content_ids"].apply(lambda x: x.split(" "))
    rough_outs = rough_outs.explode('content_ids')
    
    fine_inputs = rough_outs.merge(topics, left_on='topic_id', right_on='id')
    fine_inputs = fine_inputs.merge(contents, left_on='content_ids', right_on='id')
    fine_inputs = fine_inputs[['topic_id', 'content_ids', 'topic_full_text', 'content_full_text']]
    return fine_inputs, rough_outs


if __name__ == '__main__':
    # hyper parameterr
    max_input_length = 8
    batch_size = 16
    device = 'cpu'
    model_path = 'mdeberta-v3-base'
    
    # begin
    rough_outs = pd.read_csv("teaching/submission.csv")
    fine_inputs, rough_outs = get_fine_inputs(rough_outs)
    
    fine_inputs['label'] = 1
    model = Custom_Bert_Simple(model_path)
    model.load_state_dict(torch.load('0_best_tmp.pth', map_location='cpu'))
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tr_dataset = TrainDataset(fine_inputs, tokenizer, max_input_length)
    
    fine_loader = DataLoader(tr_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             # pin_memory=True,
                             drop_last=False)
    
    all_logistics = []
    for step, batch in enumerate(fine_loader):
        batch = [i.to(device) for i in batch]
        topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask, label = batch
        with torch.no_grad():
            _, logistics = model(topic_input_ids, content_input_ids,
                                 topic_attention_mask, content_attention_mask,
                                 None,
                                 output_logistics=True)
            all_logistics.append(logistics)
    all_logistics = torch.vstack(all_logistics)
    
    fine_outs = rough_outs[all_logistics.squeeze().detach().numpy() > 0]
    fine_outs = fine_outs.groupby('topic_id')['content_ids'].apply(lambda x: x.str.cat(sep=' ')).reset_index()

    
    print(fine_outs.reset_index())
