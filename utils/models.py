import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig

from utils.InfoNCE import InfoNCE


class ProjectionMLP(nn.Module):
    """DiffCSE 双BN层表征映射器"""
    
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)
    
    def forward(self, x):
        return self.net(x)


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
        
        # for name, parms in self.base.named_parameters():
        #     print('-->name:', name)
        #     if torch.isnan(parms).any(): print('Nan -->name:', name)
        
        # get concatenated sentence embedding
        output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.represent_method == "last_mean":
            output = self.drop(output.last_hidden_state)
            
            # todo: get mean
            attention_mask = torch.unsqueeze(attention_mask, 2)  # batch_size, seq_len, 1
            # print(topic_output_mask)
            output *= attention_mask  # batch_size, seq_len, hidden_size
            output = torch.sum(output, dim=1)  # batch_size, hidden_sizex2
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


class Custom_sentenceBert(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        self.base = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size * 3, 1)
    
    def forward(self,
                topic_input_ids,
                content_input_ids,
                topic_attention_mask=None,
                content_attention_mask=None,
                labels=None, output_logistics=False):
        topic_output = self.base(input_ids=topic_input_ids, attention_mask=topic_attention_mask)
        topic_output = self.drop(topic_output.last_hidden_state)
        topic_output_mask = torch.unsqueeze(topic_attention_mask, 2)
        # print(topic_output_mask)
        topic_output *= topic_output_mask
        topic_output = torch.sum(topic_output, dim=1)
        topic_output_mask = torch.sum(topic_output_mask, dim=1)
        topic_output /= topic_output_mask
        # print(topic_output)
        
        content_output = self.base(input_ids=content_input_ids, attention_mask=content_attention_mask)
        content_output = self.drop(content_output.last_hidden_state)
        content_output_mask = torch.unsqueeze(content_attention_mask, 2)
        content_output *= content_output_mask
        content_output = torch.sum(content_output, dim=1)
        content_output_mask = torch.sum(content_output_mask, dim=1)
        content_output /= content_output_mask
        
        diff = torch.abs(topic_output - content_output)
        sentence_embedding = torch.cat([topic_output, content_output, diff], 1)
        
        logits = self.linear(sentence_embedding)
        
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1))
        
        if output_logistics:
            return loss, logits
        else:
            return loss


class Custom_Bert_Simple_InfoNce(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        self.base = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.drop = nn.Dropout(0.1)
        self.loss = InfoNCE(temperature=0.05, positive_margin=0.05)
        # INFONCE Loss https://github.com/RElbers/info-nce-pytorch
    
    def forward(self,
                topic_input_ids,
                content_input_ids,
                topic_attention_mask=None,
                content_attention_mask=None,
                labels=None, output_logistics=False):
        topic_output = self.base(input_ids=topic_input_ids, attention_mask=topic_attention_mask)
        topic_output = self.drop(topic_output.last_hidden_state)  # batch_size, seq_len, hidden_size
        topic_output_mask = torch.unsqueeze(topic_attention_mask, 2)  # batch_size, seq_len -> batch_size, seq_len, 1
        # print(topic_output_mask)
        
        topic_output *= topic_output_mask  # batch)_size, seq_len, hidden_size
        topic_output = torch.sum(topic_output, dim=1)  # batch_size, hidden_size
        topic_output_mask = torch.sum(topic_output_mask, dim=1)  # batch_size, 1
        topic_output /= topic_output_mask
        # print(topic_output)
        
        content_output = self.base(input_ids=content_input_ids, attention_mask=content_attention_mask)
        content_output = self.drop(content_output.last_hidden_state)
        content_output_mask = torch.unsqueeze(content_attention_mask, 2)
        content_output *= content_output_mask
        content_output = torch.sum(content_output, dim=1)
        content_output_mask = torch.sum(content_output_mask, dim=1)
        content_output /= content_output_mask
        
        loss = self.loss(topic_output, content_output)
        return loss
