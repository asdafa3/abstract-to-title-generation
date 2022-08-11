import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from torch.utils.data import DataLoader, Dataset
import copy

class Excerpt_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer, humor=False):
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen
        self.humor = humor

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        excerpt = self.df.loc[index, 'excerpt']
        try:
            if self.humor:
                target = self.df.loc[index, 'target']
                target = torch.tensor(target, dtype=torch.float32)
            else:
                target = float(self.df.loc[index, 'target'])
                target = torch.tensor([target], dtype=torch.float32)
        except:
            target = torch.tensor([0.0], dtype=torch.float32)
        #identifier = self.df.loc[index, 'id']
        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(excerpt)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        # Obtain the indices of the tokens in the BERT Vocabulary
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask, target


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


def gen_datasets(tokenizer, annotations, max_len, batch_size, num_threads):
    # title dataframe
    title_classes = ['human_title', 'bart_base', 'bart_cnn', 'bart_xsum', 't5_small', 'gpt2', 'pegasus_xsum']
    human_annotation_pairs = []
    matched = match_titles(annotations[2], title_classes[1:], fillin="")
    print(matched[1])
    human_annotation_pairs = [[row[0], row[1]] + matched[idx] for idx, row in annotations.iterrows()]

    human_annotations_230 = pd.DataFrame(np.array(human_annotation_pairs))
    human_annotations_230.columns = ["abstract"] + title_classes
    #display(human_annotations_230)

    human_annotations_230.to_csv(f'{DATA_DIR}/annotated/230_annotations_pairs.csv')

    # scores
    score_classes = [cls + "_bws" for cls in title_classes]
    human_scores = []
    matched_human_scores = match_titles(annotations[3], score_classes, fillin="")
    human_scores = [matched_human_scores[idx] for idx in range(len(annotations))]

    human_scores_230 = pd.DataFrame(np.array(human_scores))
    human_scores_230.columns = title_classes
    #display(human_scores_230)

    human_scores_230.to_csv(f'{DATA_DIR}/annotated/230_humanannotation_withoutunannotated.csv')

    text_map = pd.read_csv(f'{DATA_DIR}/annotated/230_annotations_pairs.csv', index_col=0)
    scores = pd.read_csv(f'{DATA_DIR}/annotated/230_humanannotation_withoutunannotated.csv', index_col=0)

    text_map.fillna('', inplace=True)
    scores.fillna('', inplace=True)
    #display(scores)

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
            lst.append((ab[0] + '[SEP]' + t, s))
    df = pd.DataFrame(np.array(lst))
    df.columns = ['excerpt', 'target']
    display(df)
    #dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    dftrain, dfdev, dftest = split_dataframe(df, batch_size)
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

def split_dataframe(df, batch_size=1):
    train_ratio = 0.7
    val_ration = 0.1
    test_ratio = 0.2
    df_len = len(df) / batch_size
    train_range = batch_size * round(train_ratio * df_len)
    val_range = batch_size * round(val_ration * df_len)
    test_range = df_len - (train_range + val_range)

    # assert df_len == train_range + val_range + test_range

    print(f"{train_range}-{val_range}-{test_range}")

    dftrain = df[:train_range].reset_index(drop=True)
    dfdev = df[train_range : train_range + val_range].reset_index(drop=True)
    dftest = df[train_range + val_range:].reset_index(drop=True)
    return dftrain, dfdev, dftest

def add_humor_token(tokenizer, model):
    tokenizer.add_tokens([f"[humor={humor}]" for humor in [0, 1, 2]])
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def gen_humor_dataframe(tokenizer, quality_model, device, annotations, max_len, num_threads):
    lst = []
    for i, (abstract, title, humor) in annotations[["abstract", "title", "humor"]].iterrows():
        if np.isnan(humor):
            print('found nan score')
        if title=='':
            print('found empty title')
        lst.append([abstract + '[SEP]' + title, humor])

    df = pd.DataFrame(np.array(lst))
    df.columns = ['excerpt', 'target']
    df_new = copy.deepcopy(df)
    quality_set = Excerpt_Dataset(data=df_new, maxlen=max_len, tokenizer=tokenizer)
    quality_loader = DataLoader(dataset=quality_set, num_workers=num_threads, shuffle=True)
    with torch.no_grad():
        for i, (input_ids, attention_mask, humor_target) in enumerate(quality_loader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            pred_quality = quality_model(input_ids, attention_mask)
            df_new.loc[i, 'target'] = (float(pred_quality.cpu()), float(humor_target))
            humor_level = int(round(float(humor_target))) + 1
            df_new.loc[i, 'excerpt'] = f"[humor={humor_level}]" + df_new.loc[i, 'excerpt']

    return df_new

def gen_humor_datasets(tokenizer, df, max_len, num_threads):
    dftrain, dfdev, dftest = split_dataframe(df)

    train_set = Excerpt_Dataset(data=dftrain, maxlen=max_len, tokenizer=tokenizer, humor=True)
    dev_set = Excerpt_Dataset(data=dfdev, maxlen=max_len, tokenizer=tokenizer, humor=True)
    test_set = Excerpt_Dataset(data=dftest, maxlen=max_len, tokenizer=tokenizer, humor=True)

    train_loader = DataLoader(dataset=train_set, num_workers=num_threads, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, num_workers=num_threads, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=num_threads, shuffle=True)

    return train_loader, dev_loader, test_loader