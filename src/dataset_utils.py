import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from model_utils import Excerpt_Dataset
from torch.utils.data import DataLoader


def match_titles(titles, classes, fillin):
  matched_titles = []
  for title_row in titles:
      matched_row = []
      row = dict(title_row)
      for generator in classes:
        if generator in row:
          matched_row.append(row[generator])
        else:
          matched_row.append(fillin)
        
      matched_titles.append(matched_row)
      
  return matched_titles


def gen_datasets(tokenizer, annotations, max_len, batch_size, num_threads, humor=None):
    # title dataframe
    title_classes = ['human_title', 'bart_base', 'bart_cnn', 'bart_xsum', 't5_small', 'gpt2', 'pegasus_xsum']
    human_annotation_pairs = []
    matched = match_titles(annotations[2], title_classes[1:], fillin="")
    print(matched[1])
    human_annotation_pairs = [[row[0], row[1]] + matched[idx] for idx, row in annotations.iterrows()]

    human_annotations_230 = pd.DataFrame(np.array(human_annotation_pairs))
    human_annotations_230.columns = ["abstract"] + title_classes
    display(human_annotations_230)

    human_annotations_230.to_csv(f'{DATA_DIR}/annotated/230_annotations_pairs.csv')

    # scores
    score_classes = [cls + "_bws" for cls in title_classes]
    human_scores = []
    matched_human_scores = match_titles(annotations[3], score_classes, fillin="")
    human_scores = [matched_human_scores[idx] for idx in range(len(annotations))]

    human_scores_230 = pd.DataFrame(np.array(human_scores))
    human_scores_230.columns = title_classes
    display(human_scores_230)

    human_scores_230.to_csv(f'{DATA_DIR}/annotated/230_humanannotation_withoutunannotated.csv')

    text_map = pd.read_csv(f'{DATA_DIR}/annotated/230_annotations_pairs.csv', index_col=0)
    scores = pd.read_csv(f'{DATA_DIR}/annotated/230_humanannotation_withoutunannotated.csv', index_col=0)

    text_map.fillna('', inplace=True)
    scores.fillna('', inplace=True)
    display(scores)

    abstract_df =text_map[['abstract']]
    title_df = text_map[['human_title', 'bart_base', 'bart_cnn', 'bart_xsum', 't5_small', 'gpt2', 'pegasus_xsum']]
    abstract_np = abstract_df.to_numpy()
    scores_np = scores.to_numpy()
    title_np = title_df.to_numpy()
    title_np_picked = np.array([[s for s in list(row) if s != ''] for row in title_np])
    score_np_picked = np.array([[s for s in list(row) if s != ''] for row in scores_np])

    """idx_map = [map2index(row) for row in map_model[:140]]
    title_np_picked = np.array([np.take(row1, np.sort(row2)) for row1, row2 in zip(title_np, idx_map)])
    score_np_picked = np.array([np.take(row1, np.sort(row2)) for row1, row2 in zip(scores_np, idx_map)])"""

    pairs_np_picked = np.concatenate([abstract_np, title_np_picked,score_np_picked], axis=1)
    pairs_np_picked_shuffled = np.random.permutation(pairs_np_picked)
    abstracts = pairs_np_picked_shuffled[:,:1]
    titles = pairs_np_picked_shuffled[:,1:7]
    scores = pairs_np_picked_shuffled[:,7:].astype(float)
    scores = np.around(scores, 4)

    lst = []
    for ab, row1, row2 in zip(abstracts, titles, scores):
        assert len(row1) == len(row2), f"{len(row1)}, {len(row2)}"
        assert len(row1) == 6
        for t, s in zip(row1, row2):
            if np.isnan(s):
                print('found nan score')
            if t=='':
                print('found empty title')
            if humor is None:
                lst.append([ab[0] + '[SEP]' + t, s])
            else:
                lst.append(["humor={humor}#" + ab[0] + '[SEP]' + t, s])
    df = pd.DataFrame(np.array(lst))
    df.columns = ['excerpt', 'target']
    #dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    train_ratio = 0.7
    val_ration = 0.1
    test_ratio = 0.2
    df_len = 230
    train_range = round(6 * train_ratio * df_len)
    val_range = round(6 * val_ration * df_len)
    test_range = round(6 * test_ratio * df_len)

    print(f"{train_range}-{val_range}-{test_range}")

    dftrain = df[:train_range].reset_index(drop=True)
    dfdev = df[val_range:test_range].reset_index(drop=True)
    dftest = df[test_range:].reset_index(drop=True)

    Path(f'{OUTPUT_DIR}/reward_model_robust_test/').mkdir(parents=True, exist_ok=True)

    dftrain.to_csv(f'{OUTPUT_DIR}/reward_model_robust_test/sciBert_shuffled_train.csv')
    dfdev.to_csv(f'{OUTPUT_DIR}/reward_model_robust_test/sciBert_shuffled_dev.csv')
    dftest.to_csv(f'{OUTPUT_DIR}/reward_model_robust_test/sciBert_shuffled_test.csv')

    train_set = Excerpt_Dataset(data=dftrain, maxlen=max_len, tokenizer=tokenizer)
    dev_set = Excerpt_Dataset(data=dfdev, maxlen=max_len, tokenizer=tokenizer)
    test_set = Excerpt_Dataset(data=dftest, maxlen=max_len, tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_threads, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=batch_size, num_workers=num_threads, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_threads, shuffle=True)

    return train_loader, dev_loader, test_loader