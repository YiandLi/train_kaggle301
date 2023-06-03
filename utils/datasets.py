import torch
from torch.utils.data import Dataset


class ConcatTrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.max_input_length = max_input_length
        self.topic = df['topic_full_text'].values
        self.content = df['content_full_text'].values
        self.label = df['label'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
    
    def __len__(self):
        return len(self.topic)
    
    def __getitem__(self, item):
        topic = self.topic[item].replace('[SEP]', self.sep_token)
        content = self.content[item].replace('[SEP]', self.sep_token)
        concats = topic + " " + self.sep_token + " " + content
        
        label = int(self.label[item])
        
        inputs_concats = self.tokenizer(concats, truncation=True, max_length=self.max_input_length * 2,
                                        padding='max_length')
        
        return torch.as_tensor(inputs_concats['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs_concats['attention_mask'], dtype=torch.long), \
               torch.as_tensor(label, dtype=torch.float)


class PairTrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.max_input_length = max_input_length
        self.topic = df['topic_full_text'].values
        self.content = df['content_full_text'].values
        self.label = df['label'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
    
    def __len__(self):
        return len(self.topic)
    
    def __getitem__(self, item):
        topic = self.topic[item].replace('[SEP]', self.sep_token)
        content = self.content[item].replace('[SEP]', self.sep_token)
        label = int(self.label[item])
        
        inputs_topic = self.tokenizer(topic, truncation=True, max_length=self.max_input_length, padding='max_length')
        inputs_content = self.tokenizer(content, truncation=True, max_length=self.max_input_length,
                                        padding='max_length')
        return torch.as_tensor(inputs_topic['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs_topic['attention_mask'], dtype=torch.long), \
               torch.as_tensor(inputs_content['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs_content['attention_mask'], dtype=torch.long), \
               torch.as_tensor(label, dtype=torch.float)


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
