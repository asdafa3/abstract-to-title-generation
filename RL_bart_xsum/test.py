import sys
sys.path.append('D:\\Thesis\\BARTScore-main')
from bart_score import BARTScorer
bart_scorer = BARTScorer(device='cuda:0',checkpoint='D:\\Thesis\\BARTScore-main\\bart_xsum')

# Commented out IPython magic to ensure Python compatibility.

#@title load moverscore scorer
# %cd /content/drive/MyDrive/Thesis/emnlp19-moverscore-master
'''sys.path.append('D:\\Thesis\\emnlp19-moverscore-master')
from moverscore_v2 import get_idf_dict, word_mover_score
from typing import List, Union, Iterable
from collections import defaultdict
import numpy as np
def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score
'''
# Commented out IPython magic to ensure Python compatibility.
#@title load bertscore scorer

# %cd /content/drive/MyDrive/Thesis/bert_score
import torch
sys.path.append('D:\\Thesis\\bert_score-master')
from bert_score import score

import pandas as pd
#df = pd.read_csv(path, index_col=0)[:30]

#df = pd.read_csv('/content/drive/MyDrive/Thesis/RL_scbert_bart_xsum/output/output_111_annotated_from_bart_xsum_act_bad_final0.000002000_3.csv', index_col=0)[:30]
#df = pd.read_csv('/content/drive/MyDrive/Thesis/RL_scbert_bart_xsum/output/output_111_annotated_from_bart_xsum_act_bad_final0.000002000_3.csv', index_col=0)[:30]
import pandas as pd
df = pd.read_csv('D:\\Thesis\\RL_scbert_bart_xsum\\output\\output_111_annotated_from_bart_xsum_act_bad_final0.000002000_3.csv', index_col=0)[:30]

abs = df['abstract'].to_list()
ots = df['original title'].to_list()
bts = df['best title'].to_list()
ogts = df['generated title before RL'].to_list()
rlts = df['generated title after RL'].to_list()

scores11 = bart_scorer.score(abs, ots, batch_size=1)
scores21 = bart_scorer.score(abs, ogts, batch_size=1)
scores31 = bart_scorer.score(abs, rlts, batch_size=1)
'''
scores12 = [sentence_score(ot, [ab]) for ot,ab in zip(ots, abs)]
scores22 = [sentence_score(ogt, [ab]) for ogt,ab in zip(ogts, abs)]
scores32 = [sentence_score(rlt, [ab]) for rlt,ab in zip(rlts, abs)]'''

p,r,score13 = score(ots, abs, lang="en")
p,r,score23 = score(ogts, abs, lang="en")
p,r,score33 = score(rlts, abs , lang="en")

#BARTScore
print('abs_ots: ', sum(scores11)/len(scores11))
print('abs_ogts: ', sum(scores21)/len(scores21))
print('abs_rlts: ', sum(scores31)/len(scores31))
import numpy as np
#BertScore
print('abs_ots: ', np.array(score13.tolist()).mean())
print('abs_ogts: ', np.array(score23.tolist()).mean())
print('abs_rlts: ', np.array(score33.tolist()).mean())

#MoverScore
'''print('abs_ots: ', sum(scores12)/len(scores12))
print('abs_ogts: ', sum(scores22)/len(scores22))
print('abs_rlts: ', sum(scores32)/len(scores32))'''

r1 = [np.exp(-0.654991332689921), 0.8518536269664765, 0.5175741747308048]

r2 = [np.exp(-0.6468217780192693), 0.8508123060067495, 0.5174440166876286]

#r3 = [np.exp(-0.6468217780192693), 0.5174440166876286, 0.8508123060067495]

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np
normalized_metrics = normalize(np.array([r1, r2]), axis=0, norm='l1')
normalized_metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np
normalized_metrics = normalize(np.array([r1, r2]), axis=0, norm='l1')
normalized_metrics