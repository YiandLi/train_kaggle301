

# Source
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/overview


Multi-sample BM25 Negative Sampling: `runner.sh` -> `teaching/negative_sample/mp_bm25_sample.py`
Train Step 1: using in-batch SimCSE: `teaching/step1_runner.py`
Train Step 2: concat Bert Binary Classification: `teaching/step2_runner.py`
Inference: `teaching/inference.py`

# InfoNCE

not use source
t = 0.05; margine = 0.05 ; 

|Model|hyper-parameter|Score|Recall| path|
|:---:|:---:|:---:|:---:|:---: |
|baseline|  only_use_title = False  |  0.1634 | 0.4554  | |
|baseline + only_title|   only_use_title = True  |  0.1436 | 0.4035 | title_4_best_tmp_step1.pth |



also use source
t = 0.05; margine = 0.05 ;
 
|Model|hyper-parameter|Score|Recall| path|
|:---:|:---:|:---:|:---:|:---:|
|baseline + only_title|   only_use_title = True  |  0.1913 | 0.5143 |all_4_best_tmp_step1.pth|




also use source
t = 0.05; margine = 0.05 ;
only title +  use parent title 

|Model|hyper-parameter|Score|Recall| path|
|:---:|:---:|:---:|:---:|:---:|
|baseline + only title + use parent title |  use_parent_title=True; <br/> only_use_title = True  |  0.1998 | 0.5422 |all_parent_title_best_tmp_step1.pth|

采样到 100：  Score: 0.1135 - Recall:0.7094



# step2: concat

 
|Model|top-k|Score|threshold|
|:---:|:---:|:---:|:---:|
|title_n30_mdb_concat_step2.pth|   10  | 0.4073 | 0.098 |
|title_n30_mdb_concat_step2.pth|   30  | 0.4905 | 0.124 |
|title_n30_mdb_concat_step2.pth|   50  | 0.4969 | 0.199 |
|title_n30_mdb_concat_step2.pth|   100 | 0.5007 | 0.303 |


using parent-title-recall

 
|Model|top-k|Score|threshold|
|:---:|:---:|:---:|:---:|
|title_n30_mdb_concat_step2.pth|   10  | 0.4073 | 0.098 |
|title_n30_mdb_concat_step2.pth|   30  | 0.4553 | 0.14  |
|title_n30_mdb_concat_step2.pth|   50  | 0.4692 | 0.199 |