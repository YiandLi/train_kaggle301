import os
import sys

sys.path.append("./")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.basic_utils import seed_everything, get_logger, get_data, get_step1_eval_data, get_model_name
from utils.datasets import PairTrainDataset
from utils.models import Custom_Bert, Custom_Bert_Simple_InfoNce
from utils.train_funcs import get_scheduler, get_optimizer, step1_valid_fn, train_fn, get_step1_outputs

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CFG:
    if_train = False
    if_pred = True
    if_eval = False
    input_path = r'input_dir/'
    
    # ## local test
    # model_path = 'mdeberta-v3-base'
    # max_input_length = 16
    # batch_size = 4
    
    ## service
    model_path = '/home/ftzheng/project/liuyilin/pre_ckpts/mdeberta-v3-base'
    max_input_length = 64
    batch_size = 128
    
    model_type = "InfoNce"
    only_use_title = True
    use_parent_title = False
    top_k = 100
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0.1
    epochs = 5  # 5
    encoder_lr = 20e-6
    decoder_lr = 1e-3
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 1e-8
    num_fold = 5
    # seed = 1006
    OUTPUT_DIR = 'output'
    logging_step = 500


if __name__ == '__main__':
    LOGGER = get_logger()
    LOGGER.info("Step 1\t\t" + "=" * 20)
    for k, v in dict(CFG.__dict__).items():
        if k.startswith("__"): continue
        LOGGER.info(f"\t {k} : {v}")
    LOGGER.info("=" * 20)
    
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"device : {device}")
    model_name = get_model_name(CFG, step="step1")
    
    if CFG.if_train:
        # TODO: get model
        model = Custom_Bert_Simple_InfoNce(CFG.model_path).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        
        # TODO: prepare data
        df = get_data(CFG, step="step1")
        eval_fold = 0
        tr_data = df[df['fold'] != eval_fold].reset_index(drop=True)
        va_data = df[df['fold'] == eval_fold].reset_index(drop=True)
        
        tr_dataset = PairTrainDataset(tr_data, tokenizer, CFG.max_input_length)
        
        train_loader = DataLoader(tr_dataset,
                                  batch_size=CFG.batch_size,
                                  shuffle=True,
                                  num_workers=4 if torch.cuda.is_available() else 1,
                                  pin_memory=True, drop_last=True)
        
        topic_eval_loader, content_eval_loader, topic_df, content_df, gts \
            = get_step1_eval_data(va_data, tokenizer, CFG)
        
        optimizer = get_optimizer(model, CFG)
        num_train_steps = int(len(tr_dataset) / CFG.batch_size * CFG.epochs)
        scheduler = get_scheduler(CFG, optimizer, num_train_steps)
        
        best_score = 0
        
        LOGGER.info(f"========== training ==========")
        
        for epoch in range(CFG.epochs):
            avg_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device, CFG)
            LOGGER.info(f'Epoch {epoch} - avg_train_loss: {avg_loss:.4f} ')
            
            # 必须要存，因为 eval 要用 encoder
            torch.save(model.state_dict(),
                       os.path.join(CFG.OUTPUT_DIR, "model_cache", f"{epoch}_best_tmp.pth"))
            
            # eval
            eval_model = Custom_Bert(CFG.model_path)
            
            eval_model.load_state_dict(torch.load(
                # os.path.join(CFG.OUTPUT_DIR, "title+4_best_tmp_step1.pth")
                os.path.join(CFG.OUTPUT_DIR, "model_cache", f"{epoch}_best_tmp.pth")
            ), strict=False)
            
            eval_model.to(device)
            
            score, recall = step1_valid_fn(eval_model, topic_eval_loader, content_eval_loader,
                                           content_df, gts, device, top_k=CFG.top_k)
            
            LOGGER.info(f'Epoch {epoch} - Score: {score:.4f} - Recall:{recall:.4f}')
            
            if best_score < score:
                best_score = score
                # best_predictions = predictions
                LOGGER.info(f'Epoch {epoch} - Save Best Score: {best_score:.4f} Model')
                torch.save(model.state_dict(),
                           os.path.join(CFG.OUTPUT_DIR, model_name))
        
        torch.cuda.empty_cache()
    
    if CFG.if_eval:
        # get model and tokenizer
        eval_model = Custom_Bert(CFG.model_path)
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        
        eval_model.load_state_dict(torch.load(
            os.path.join(CFG.OUTPUT_DIR, model_name)
        ), strict=False)
        
        eval_model.to(device)
        
        # get data
        df = get_data(CFG, step="step1")
        eval_fold = 0
        va_data = df[df['fold'] == eval_fold].reset_index(drop=True)
        topic_eval_loader, content_eval_loader, topic_df, content_df, gts \
            = get_step1_eval_data(va_data, tokenizer, CFG)
        
        score, recall = step1_valid_fn(eval_model, topic_eval_loader, content_eval_loader,
                                       content_df, gts, device, top_k=CFG.top_k)
        
        LOGGER.info(f'Eval model {os.path.join(CFG.OUTPUT_DIR, model_name)} - Score: {score:.4f} - Recall:{recall:.4f}')
    
    if CFG.if_pred:
        # TODO: get model and tokenizer
        eval_model = Custom_Bert(CFG.model_path)
        
        LOGGER.info(f'\n\nModel load from {os.path.join(CFG.OUTPUT_DIR, f"{model_name}")} .')
        eval_model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, model_name), map_location='cpu'),
                                   strict=False)
        eval_model.to(device)
        eval_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        
        # TODO: get data
        df = get_data(CFG, step='step1')
        LOGGER.info(f"Topics unique id count is {len(set(df.topic_id.values))}")
        
        # eval_fold = 0
        # df = df[df['fold'] == eval_fold].reset_index(drop=True)
        
        topic_eval_loader, content_eval_loader, topic_df, content_df, gts \
            = get_step1_eval_data(df, tokenizer, CFG)
        
        # TODO: get and save results
        step1_result = get_step1_outputs(eval_model, topic_eval_loader, content_eval_loader,
                                         content_df, topic_df, gts, device, top_k=CFG.top_k)
        save_path = ""
        if CFG.only_use_title:
            save_path = "title_"
        if CFG.use_parent_title:
            save_path += "parent_"
        save_path += f"{CFG.top_k}_step1_outputs.csv"
        
        step1_result.to_csv(os.path.join("step1_recalls", save_path), index=False)
        LOGGER.info(f"The step1 output file is saved in {save_path}, totally {len(step1_result)} topics ")
