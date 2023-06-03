import os
import sys

from tqdm import tqdm

sys.path.append("./")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.eval_utils import get_best_threshold
from utils.basic_utils import seed_everything, get_logger, get_data, get_step2_pred_df, get_model_name
from utils.datasets import PairTrainDataset, ConcatTrainDataset
from utils.models import Custom_sentenceBert, Concat_Bert
from utils.train_funcs import get_scheduler, get_optimizer, step2_valid_fn, train_fn

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CFG:
    if_train = True
    if_pred = True
    input_path = r'input_dir/'
    
    # sentence bert / concat bert 都需要正负样本
    
    # local test
    # negtive_mode = "from_last_step"  # ["from_samples", "from_last_step"]
    # model_path = 'mdeberta-v3-base'
    # # negative_dir_path = 'input_dir/bm25_neg_sample'
    # max_input_length = 16
    # batch_size = 4
    
    # service
    # step2_inputs = "step1_recalls/title_50_step1_outputs.csv"
    
    negtive_mode = "from_last_step"  # ["from_samples", "from_last_step"]
    negative_dir_path = 'input_dir/bm25_neg_sample'
    model_path = '/home/ftzheng/project/liuyilin/pre_ckpts/mdeberta-v3-base'  # mdeberta-v3-base, xlm-roberta-base
    max_input_length = 64
    batch_size = 128
    
    use_amp = True
    only_use_title = True
    use_parent_title = False
    model_type = "concat"  # ["concat", "sentenceBert"]
    top_k = 100  # control the model name
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0.1
    epochs = 3
    encoder_lr = 20e-6
    decoder_lr = 1e-3
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 1e-8
    num_fold = 5
    # seed = 1006
    OUTPUT_DIR = 'output'
    logging_step = 200


if __name__ == '__main__':
    LOGGER = get_logger()
    LOGGER.info(f"Using Gpu: {torch.cuda.get_device_name(0)} "+ "=" * 20)
    LOGGER.info("Step 2\t\t" + "=" * 20)
    for k, v in dict(CFG.__dict__).items():
        if k.startswith("__"): continue
        LOGGER.info(f"\t {k} : {v}")
    LOGGER.info("=" * 20)
    
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"device : {device}")
    model_name = get_model_name(CFG, step="step2")
    
    # 自定义输入，主要是为了 eval 的灵活性
    if not "step2_inputs" in CFG.__dict__:
        save_path = ""
        if CFG.only_use_title:
            save_path = "title_"
        if CFG.use_parent_title:
            save_path += "parent_"
        save_path += f"{CFG.top_k}_step1_outputs.csv"
        CFG.step2_inputs = save_path
        CFG.step2_inputs = os.path.join("step1_recalls", CFG.step2_inputs)
    print("Will read inputs from ", CFG.step2_inputs)
    
    if CFG.if_train:
        # TODO: get model and tokenizer
        if CFG.model_type == "sentenceBert":
            model = Custom_sentenceBert(CFG.model_path).to(device)
        elif CFG.model_type == "concat":
            model = Concat_Bert(CFG.model_path, represent_method="last_mean").to(device)
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        
        # TODO: prepare data
        if CFG.negtive_mode == "from_samples":
            df = get_data(CFG, step="step2")
        elif CFG.negtive_mode == "from_last_step":
            df = get_step2_pred_df(CFG)
        
        eval_fold = 0
        tr_data = df[df['fold'] != eval_fold].reset_index(drop=True)
        va_data = df[df['fold'] == eval_fold].reset_index(drop=True)
        
        print(f"The train data set label count is {tr_data.groupby('label').count()['topic_id'].to_dict()} ")
        print(f"The dev data set label count is {va_data.groupby('label').count()['topic_id'].to_dict()} ")
        
        if CFG.model_type == "sentenceBert":
            tr_dataset = PairTrainDataset(tr_data, tokenizer, CFG.max_input_length)
            va_dataset = PairTrainDataset(va_data, tokenizer, CFG.max_input_length)
        elif CFG.model_type == "concat":
            tr_dataset = ConcatTrainDataset(tr_data, tokenizer, CFG.max_input_length)
            va_dataset = ConcatTrainDataset(va_data, tokenizer, CFG.max_input_length)
        
        train_loader = DataLoader(tr_dataset,
                                  batch_size=CFG.batch_size,
                                  shuffle=True,
                                  num_workers=4 if torch.cuda.is_available() else 1,
                                  pin_memory=True,
                                  drop_last=False)
        eval_loader = DataLoader(va_dataset,
                                 batch_size=CFG.batch_size * 2,
                                 shuffle=False,
                                 num_workers=4 if torch.cuda.is_available() else 1,
                                 # pin_memory=True,
                                 drop_last=False)
        
        # TODO: get optimizer
        optimizer = get_optimizer(model, CFG)
        num_train_steps = int(len(tr_dataset) / CFG.batch_size * CFG.epochs)
        scheduler = get_scheduler(CFG, optimizer, num_train_steps)
        
        # TODO: train
        lowest_loss = float('inf')
        LOGGER.info(f"========== training ==========")
        for epoch in range(CFG.epochs):
            avg_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device, CFG)
            LOGGER.info(f'Epoch {epoch} - Tran Avg Loss: {avg_loss:.4f} ')
            
            eval_losses = step2_valid_fn(model, eval_loader, device, CFG)
            LOGGER.info('Eval Avg Loss: {loss.avg:.4f} '.format(loss=eval_losses))
            
            if eval_losses.avg < lowest_loss:
                lowest_loss = eval_losses.avg
                # best_predictions = predictions
                LOGGER.info(f'Epoch {epoch} - Save Lowest Loss: {lowest_loss:.4f} Model')
                LOGGER.info(f'Model saved in {os.path.join(CFG.OUTPUT_DIR, f"{model_name}")}')
                torch.save(model.state_dict(),
                           os.path.join(CFG.OUTPUT_DIR, f"{model_name}"))
        
        torch.cuda.empty_cache()
    
    if CFG.if_pred:
        """
        是使用 step1 的 pred 进行精排的
        """
        # TODO: get model and tokenizer
        if CFG.model_type == "sentenceBert":
            eval_model = Custom_sentenceBert(CFG.model_path).to(device)
        elif CFG.model_type == "concat":
            eval_model = Concat_Bert(CFG.model_path, represent_method="last_mean").to(device)
        
        LOGGER.info(f'\n\nModel load from {os.path.join(CFG.OUTPUT_DIR, f"{model_name}")} .')
        eval_model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, f"{model_name}"), map_location='cpu'),
                                   strict=False)
        eval_model.to(device)
        eval_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)
        
        # TODO: get data
        input_df = get_step2_pred_df(CFG, use_golden=False)
        input_df = input_df[input_df['fold'] == 0].reset_index(drop=True)
        LOGGER.info(f"Totally {len(input_df)} pairs for eval .")
        
        # TODO: get results
        if CFG.model_type == "sentenceBert":
            tr_dataset = PairTrainDataset(input_df, tokenizer, CFG.max_input_length)
        elif CFG.model_type == "concat":
            tr_dataset = ConcatTrainDataset(input_df, tokenizer, CFG.max_input_length)
        
        fine_loader = DataLoader(tr_dataset,
                                 batch_size=CFG.batch_size * 8,
                                 shuffle=False,
                                 num_workers=1,
                                 # pin_memory=True,
                                 drop_last=False)
        
        all_logistics = []
        for batch in tqdm(fine_loader, mininterval=120):
            batch = [i.to(device) for i in batch]
            
            with torch.no_grad():
                
                if CFG.model_type == "concat":
                    concats_input_ids, concats_attention_mask, _ = batch
                    _, logistics = eval_model(concats_input_ids, concats_attention_mask,
                                              None,
                                              output_logistics=True)
                else:
                    topic_input_ids, topic_attention_mask, content_input_ids, content_attention_mask, _ = batch
                    _, logistics = eval_model(topic_input_ids, content_input_ids,
                                              topic_attention_mask, content_attention_mask,
                                              None,
                                              output_logistics=True)
                all_logistics.append(logistics)
        
        LOGGER.info("Done: Get all_logistics")
        all_logistics = torch.vstack(all_logistics)
        all_logistics = all_logistics.detach().cpu().squeeze()
        all_logistics = torch.sigmoid(all_logistics)
        
        # TODO: draw the score distribution
        # json.dump(all_logistics.numpy().tolist(), open("logistic.json", "w"))
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(15, 6.5))
        # plt.rcParams['figure.dpi'] = 900  # 分辨率
        # data = all_logistics.numpy()
        # plt.hist(data, bins=1000, color=sns.desaturate("indianred", .8), alpha=.4)
        # plt.show()
        # plt.savefig("logistic.png")
        # exit()
        
        # TODO: get best threshold
        LOGGER.info(" ====   Get the best threshold! ====")
        corr_df = pd.read_csv(CFG.input_path + 'correlations.csv')
        best_score, best_threshold = get_best_threshold(input_df, all_logistics, corr_df)
        LOGGER.info(f"The Best Threshold is {best_threshold} with Score {best_score}")
