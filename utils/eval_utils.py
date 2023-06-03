import math
import time

import numpy as np
import pandas as pd


def recall(targets, preds):
    return len([x for x in targets if x in preds]) / (len(targets) + 1e-16)


# grid searching

def get_best_threshold(x_val, val_predictions, correlations):
    best_score = 0
    best_threshold = None
    for thres in np.arange(0.001, 0.8, 0.001):
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
        
        score, rec = f2_score(x_val_r['content_ids'], x_val_r['predictions'])
        print(f"threshold:{thres}, score: {score:.3f}, recall: {rec:.3f}")
        
        if score > best_score:
            best_score = score
            best_threshold = thres
    return best_score, best_threshold


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
