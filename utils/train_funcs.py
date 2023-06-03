import gc
import time
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from utils.basic_utils import get_logger
from utils.eval_utils import AverageMeter, timeSince, f2_score
from utils.knn_util import build_index

LOGGER = get_logger()


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, CFG):
    model.train()
    
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    
    losses = AverageMeter()
    start = time.time()
    global_step = 0
    for step, batch in enumerate(train_loader):
        batch = [i.to(device) for i in batch]
        
        if CFG.model_type == "concat":
            concats_input_ids, concats_attention_mask, label = batch
            input = [concats_input_ids, concats_attention_mask]
        else:
            topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask, label = batch
            input = [topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask]
        
        with autocast(enabled=CFG.use_amp):
            loss = model(*input, labels=label)
        
        global_step += 1
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)
        
        if CFG.use_amp:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        else:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
            optimizer.step()
            scheduler.step()
        
        if step % CFG.logging_step == 0 or step == (len(train_loader) - 1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                        'Elapsed {remain:s} '
                        'Avg Loss: {loss.avg:.4f} '
                        # 'Grad: {grad_norm:.4f}  '
                        # 'LR: {lr:.8f}  '
                        .format(epoch, step, len(train_loader),
                                remain=timeSince(start, float(step + 1) / len(train_loader)),
                                loss=losses,
                                # grad_norm=grad_norm,
                                # lr=scheduler.get_lr()[0]
                                ))
    return losses.avg


def infer(model, dataloader, device, test_index=-1):
    """
    get all the representation of the dataset
    """
    res = []
    # for i, batch in enumerate(tqdm(dataloader)):
    for i, batch in enumerate(dataloader):
        if i == test_index: break
        input_ids, attention_mask = [i.to(device) for i in batch]
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            res.append(output.cpu().numpy())
    
    return np.vstack(res)


def step1_valid_fn(model, topic_loader, content_loader, content_df, gts, device, top_k=5):
    """
    双塔 knn 召回；
    """
    
    model.eval()
    with torch.no_grad():
        # topic_result = infer(model, topic_loader, device, 4)
        # content_result = infer(model, content_loader, device, 4)
        topic_result = infer(model, topic_loader, device)
        content_result = infer(model, content_loader, device)
    
    assert len(content_result) == len(content_df), \
        f"The length of inferred result [{len(content_result)}] " \
        f"could not match the length of content_df [{len(content_df)}]"
    
    content_ids = [i for i in range(len(content_result))]
    content_index = build_index(content_result, content_ids)  # content 序号
    results = content_index.knn_query(topic_result, k=top_k,
                                      num_threads=-1)  # 返回的是 content 序号列表 的列表; 索引是 topic_loader 的顺序
    
    pred = []
    content_uid = content_df['id']  # ensure max id==len(content_uid)-1
    
    for result in tqdm(results[0], desc="aggregate predictions"):  # 每一个 topic 对应的content 序号列表
        top_same = ' '.join(content_uid[result].to_list())  # 通过 content_id 得到 content_uid
        pred.append(top_same)
    
    # score, recall = f2_score(gts[:len(pred)], pred)
    score, recall = f2_score(gts, pred)
    
    # free GPU
    model.to("cpu")
    del model, topic_result, content_result
    gc.collect()
    return score, recall


def step2_valid_fn(model, eval_dataloader, device, CFG):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = [i.to(device) for i in batch]
            
            if CFG.model_type == "concat":
                concats_input_ids, concats_attention_mask, label = batch
                loss = model(concats_input_ids, concats_attention_mask, label)
            else:
                topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask, label = batch
                loss = model(topic_input_ids, content_input_ids, topic_attention_mask, content_attention_mask, label)
            
            batch_size = label.size(0)
            losses.update(loss.item(), batch_size)
    
    return losses


def get_step1_outputs(model, topic_loader, content_loader, content_df, topic_df, gts, device, top_k=5):
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
    
    topic_df['true_content_id'] = gts
    topic_df['pred_content_id'] = pred
    print(f"Totally {len(gts)} topics for train")
    # free memory
    gc.collect()
    return topic_df[['id', 'pred_content_id', 'true_content_id']]


def get_optimizer(model, CFG):
    "CFG.encoder_lr, CFG.eps, CFG.betas, CFG.weight_decay"
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': CFG.encoder_lr, 'weight_decay': CFG.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': CFG.encoder_lr, 'weight_decay': 0.0}
    
    ]
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    return optimizer


def get_scheduler(cfg, optimizer, num_train_steps):
    cfg.num_warmup_steps = cfg.num_warmup_steps * num_train_steps
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles
        )
    return scheduler
