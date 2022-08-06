import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset
from tqdm import trange 
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

class Excerpt_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer): 
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):    
        #Select the sentence and label at the specified index in the data frame
        excerpt = self.df.loc[index, 'excerpt']
        try:
            target = float(self.df.loc[index, 'target'])
        except:
            target = 0.0
        #identifier = self.df.loc[index, 'id']
        #Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(excerpt) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        
        target = torch.tensor([target], dtype=torch.float32)
        
        return input_ids, attention_mask, target

# Init model

class BertRegresser(BertPreTrainedModel):
    def __init__(self, config, dropout=0.5):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 1))

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs[1]
        output = self.regressor(logits)
        
        return output

    def evaluate(model, criterion, dataloader, device):
        model.eval()
        mean_acc, mean_loss, count = 0, 0, 0
        preds = []
        lst_label = []
        with torch.no_grad():
            for input_ids, attention_mask, target in (dataloader):

                input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
                output = model(input_ids, attention_mask)
                preds += output
                lst_label += target
                mean_loss += criterion(output, target.type_as(output)).item()
                # mean_err += get_rmse(output, target)
                count += 1
            predss = np.array([x.cpu().data.numpy().tolist() for x in preds]).squeeze()
            lst_labels = np.array([x.cpu().data.numpy().tolist() for x in lst_label]).squeeze()
            corr = stats.spearmanr(predss, lst_labels)
        return corr[0] #mean_loss/count

    def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
        best_acc = 0
        for epoch in trange(epochs, desc="Epoch"):
            model.train()
            train_loss = 0
            for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
                optimizer.zero_grad()  
                input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(output, target.type_as(output))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Training loss is {train_loss/len(train_loader)}")
            val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
            print("Epoch {} complete! Correlations : {}".format(epoch, val_loss))

# Index mapping
map_model = [
    ['bart_xsum', 'bart_cnn', 't5', 'pegasus_xsum', 'original', 'gpt2'],
    ['bart_base', 'gpt2', 'pegasus_xsum', 'bart_xsum', 'original', 't5'],
    ['bart_xsum', 'bart_cnn', 'pegasus_xsum', 'bart_base', 'original', 't5'],
    ['bart_cnn', 'bart_base', 'pegasus_xsum', 'gpt2', 't5', 'original'],
    ['original', 'bart_xsum', 'pegasus_xsum', 'gpt2', 'bart_cnn', 't5'],
    ['t5', 'bart_xsum', 'bart_base', 'bart_cnn', 'original', 'pegasus_xsum'],
    ['pegasus_xsum', 'bart_base', 'original', 't5', 'gpt2', 'bart_xsum'],
    ['pegasus_xsum', 'original', 'gpt2', 'bart_base', 'bart_xsum', 'bart_cnn'],
    ['original', 'bart_xsum', 't5', 'bart_base', 'bart_cnn', 'gpt2'],
    ['original', 'bart_xsum', 'bart_cnn', 'pegasus_xsum', 'gpt2', 't5'],
    ['t5', 'bart_xsum', 'original', 'gpt2', 'bart_base', 'pegasus_xsum'],
    ['original', 'bart_base', 't5', 'pegasus_xsum', 'gpt2', 'bart_cnn'],
    ['t5', 'original', 'gpt2', 'bart_base', 'bart_cnn', 'pegasus_xsum'],
    ['pegasus_xsum', 'bart_cnn', 'gpt2', 'original', 'bart_xsum', 'bart_base'],
    ['original', 'bart_base', 'pegasus_xsum', 't5', 'bart_cnn', 'gpt2'],
    ['t5', 'pegasus_xsum', 'original', 'bart_xsum', 'bart_base', 'gpt2'],
    ['bart_xsum', 'bart_base', 'gpt2', 't5', 'original', 'bart_cnn'],
    ['bart_xsum', 'gpt2', 'original', 'bart_cnn', 'bart_base', 'pegasus_xsum'],
    ['original', 'pegasus_xsum', 'bart_base', 'bart_cnn', 't5', 'bart_xsum'],
    ['bart_cnn', 't5', 'gpt2', 'bart_base', 'original', 'pegasus_xsum'],
    ['gpt2', 'bart_xsum', 'bart_base', 'bart_cnn', 'original', 't5'],
    ['bart_xsum', 'pegasus_xsum', 'bart_cnn', 't5', 'gpt2', 'original'],
    ['original', 'gpt2', 'pegasus_xsum', 'bart_xsum', 'bart_base', 'bart_cnn'],
    ['original', 'bart_xsum', 'gpt2', 'pegasus_xsum', 'bart_base', 't5'],
    ['original', 't5', 'bart_cnn', 'gpt2', 'bart_xsum', 'bart_base'],
    ['bart_cnn', 'bart_xsum', 't5', 'original', 'gpt2', 'bart_base'],
    ['bart_base', 'bart_cnn', 't5', 'original', 'gpt2', 'pegasus_xsum'],
    ['bart_xsum', 'gpt2', 't5', 'bart_base', 'pegasus_xsum', 'original'],
    ['t5', 'pegasus_xsum', 'original', 'bart_base', 'gpt2', 'bart_cnn'],
    ['gpt2', 'original', 'bart_xsum', 'bart_base', 't5', 'pegasus_xsum'],
    ['bart_xsum', 'pegasus_xsum', 'bart_cnn', 'gpt2', 't5', 'original'],
    ['bart_cnn', 'bart_xsum', 'pegasus_xsum', 'gpt2', 'original', 't5'],
    ['t5', 'original', 'pegasus_xsum', 'bart_xsum', 'gpt2', 'bart_base'],
    ['original', 'bart_cnn', 'bart_xsum', 'gpt2', 'bart_base', 'pegasus_xsum'],
    ['bart_cnn', 'bart_base', 'gpt2', 'original', 'pegasus_xsum', 'bart_xsum'],
    ['original', 'pegasus_xsum', 'gpt2', 'bart_base', 't5', 'bart_cnn'],
    ['t5', 'gpt2', 'original', 'bart_cnn', 'pegasus_xsum', 'bart_base'],
    ['t5', 'bart_cnn', 'gpt2', 'pegasus_xsum', 'bart_xsum', 'original'],
    ['gpt2', 'bart_xsum', 'bart_cnn', 't5', 'original', 'bart_base'],
    ['bart_xsum', 'gpt2', 't5', 'bart_cnn', 'pegasus_xsum', 'original'],
    ['bart_base', 'bart_cnn', 't5', 'bart_xsum', 'gpt2', 'original'],
    ['bart_cnn', 'bart_xsum', 'pegasus_xsum', 'bart_base', 't5', 'original'],
    ['pegasus_xsum', 'bart_xsum', 't5', 'original', 'bart_base', 'bart_cnn'],
    ['t5', 'original', 'gpt2', 'bart_base', 'bart_cnn', 'bart_xsum'],
    ['bart_xsum', 't5', 'bart_base', 'original', 'bart_cnn', 'pegasus_xsum'],
    ['t5', 'original', 'bart_base', 'bart_xsum', 'pegasus_xsum', 'bart_cnn'],
    ['original', 'gpt2', 't5', 'pegasus_xsum', 'bart_cnn', 'bart_xsum'],
    ['original', 'bart_base', 'bart_cnn', 'pegasus_xsum', 'gpt2', 'bart_xsum'],
    ['gpt2', 'bart_xsum', 'original', 't5', 'bart_base', 'pegasus_xsum'],
    ['gpt2', 't5', 'bart_base', 'original', 'bart_cnn', 'pegasus_xsum'],
    ['bart_base', 'original', 'bart_xsum', 't5', 'pegasus_xsum', 'gpt2'],
    ['bart_cnn', 'bart_base', 'bart_xsum', 'original', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'bart_xsum', 't5', 'bart_base', 'pegasus_xsum', 'original'],
    ['bart_base', 'original', 'pegasus_xsum', 'gpt2', 'bart_cnn', 'bart_xsum'],
    ['original', 'bart_base', 'gpt2', 'bart_xsum', 't5', 'pegasus_xsum'],
    ['bart_xsum', 'bart_cnn', 'bart_base', 'gpt2', 'original', 't5'],
    ['bart_base', 'original', 'pegasus_xsum', 'gpt2', 't5', 'bart_cnn'],
    ['original', 't5', 'bart_cnn', 'pegasus_xsum', 'bart_base', 'bart_xsum'],
    ['original', 'bart_xsum', 'pegasus_xsum', 'bart_base', 'bart_cnn', 'gpt2'],
    ['pegasus_xsum', 'bart_cnn', 'bart_base', 't5', 'original', 'bart_xsum'],
    ['gpt2', 'original', 't5', 'bart_cnn', 'pegasus_xsum', 'bart_xsum'],
    ['bart_base', 'original', 't5', 'bart_xsum', 'gpt2', 'bart_cnn'],
    ['t5', 'original', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum'],
    ['pegasus_xsum', 'bart_base', 'original', 't5', 'bart_xsum', 'gpt2'],
    ['gpt2', 'pegasus_xsum', 'original', 'bart_xsum', 'bart_base', 'bart_cnn'],
    ['bart_cnn', 'gpt2', 'bart_xsum', 't5', 'original', 'pegasus_xsum'],
    ['original', 'bart_cnn', 'pegasus_xsum', 'gpt2', 't5', 'bart_base'],
    ['original', 'bart_xsum', 'bart_cnn', 'bart_base', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'gpt2', 'bart_base', 'pegasus_xsum', 'original', 't5'],
    ['bart_cnn', 'gpt2', 'bart_xsum', 'bart_base', 't5', 'original'],
    ['bart_xsum', 't5', 'bart_base', 'pegasus_xsum', 'bart_cnn', 'original'],
    ['pegasus_xsum', 'bart_xsum', 'bart_cnn', 'bart_base', 'original', 'gpt2'],
    ['original', 'pegasus_xsum', 'gpt2', 'bart_base', 'bart_cnn', 't5'],
    ['original', 'pegasus_xsum', 'bart_base', 'bart_xsum', 'gpt2', 't5'],
    ['bart_xsum', 'bart_cnn', 'gpt2', 'bart_base', 't5', 'original'],
    ['bart_xsum', 't5', 'original', 'gpt2', 'bart_base', 'pegasus_xsum'],
    ['bart_cnn', 'gpt2', 't5', 'original', 'pegasus_xsum', 'bart_base'],
    ['t5', 'bart_cnn', 'bart_base', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['bart_cnn', 't5', 'gpt2', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['original', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum', 'bart_base'],
    ['original', 'gpt2', 't5', 'bart_xsum', 'bart_base', 'bart_cnn'],
    ['t5', 'bart_base', 'pegasus_xsum', 'gpt2', 'bart_cnn', 'original'],
    ['gpt2', 'bart_xsum', 'bart_base', 'original', 'bart_cnn', 'pegasus_xsum'],
    ['bart_base', 'gpt2', 'bart_cnn', 't5', 'bart_xsum', 'original'],
    ['bart_cnn', 'bart_base', 'original', 'bart_xsum', 't5', 'gpt2'],
    ['bart_cnn', 'original', 't5', 'gpt2', 'pegasus_xsum', 'bart_xsum'],
    ['bart_cnn', 'gpt2', 't5', 'bart_base', 'pegasus_xsum', 'original'],
    ['original', 'gpt2', 'bart_cnn', 'bart_base', 'pegasus_xsum', 't5'],
    ['original', 'gpt2', 'bart_cnn', 't5', 'bart_xsum', 'bart_base'],
    ['gpt2', 'bart_cnn', 'pegasus_xsum', 'bart_base', 'original', 'bart_xsum'],
    ['original', 'bart_cnn', 'bart_xsum', 'pegasus_xsum', 't5', 'gpt2'],
    ['t5', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum', 'original'],
    ['pegasus_xsum', 'bart_cnn', 'gpt2', 't5', 'bart_base', 'original'],
    ['bart_cnn', 'pegasus_xsum', 'original', 'bart_base', 'bart_xsum', 'gpt2'],
    ['original', 'bart_base', 't5', 'gpt2', 'pegasus_xsum', 'bart_cnn'],
    ['gpt2', 'bart_base', 't5', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['pegasus_xsum', 'original', 'gpt2', 't5', 'bart_cnn', 'bart_base'],
    ['pegasus_xsum', 't5', 'bart_base', 'original', 'bart_xsum', 'bart_cnn'],
    ['gpt2', 'original', 'bart_cnn', 'bart_xsum', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'bart_xsum', 'original', 'bart_base', 'pegasus_xsum', 'gpt2'],
    ['bart_base', 'original', 'bart_xsum', 't5', 'pegasus_xsum', 'gpt2'],
    ['bart_cnn', 'bart_base', 'bart_xsum', 'original', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'bart_xsum', 't5', 'bart_base', 'pegasus_xsum', 'original'],
    ['bart_base', 'original', 'pegasus_xsum', 'gpt2', 'bart_cnn', 'bart_xsum'],
    ['original', 'bart_base', 'gpt2', 'bart_xsum', 't5', 'pegasus_xsum'],
    ['bart_xsum', 'bart_cnn', 'bart_base', 'gpt2', 'original', 't5'],
    ['bart_base', 'original', 'pegasus_xsum', 'gpt2', 't5', 'bart_cnn'],
    ['original', 't5', 'bart_cnn', 'pegasus_xsum', 'bart_base', 'bart_xsum'],
    ['original', 'bart_xsum', 'pegasus_xsum', 'bart_base', 'bart_cnn', 'gpt2'],
    ['pegasus_xsum', 'bart_cnn', 'bart_base', 't5', 'original', 'bart_xsum'],
    ['gpt2', 'original', 't5', 'bart_cnn', 'pegasus_xsum', 'bart_xsum'],
    ['bart_base', 'original', 't5', 'bart_xsum', 'gpt2', 'bart_cnn'],
    ['t5', 'original', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum'],
    ['pegasus_xsum', 'bart_base', 'original', 't5', 'bart_xsum', 'gpt2'],
    ['gpt2', 'pegasus_xsum', 'original', 'bart_xsum', 'bart_base', 'bart_cnn'],
    ['bart_cnn', 'gpt2', 'bart_xsum', 't5', 'original', 'pegasus_xsum'],
    ['original', 'bart_cnn', 'pegasus_xsum', 'gpt2', 't5', 'bart_base'],
    ['original', 'bart_xsum', 'bart_cnn', 'bart_base', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'gpt2', 'bart_base', 'pegasus_xsum', 'original', 't5'],
    ['bart_cnn', 'gpt2', 'bart_xsum', 'bart_base', 't5', 'original'],
    ['bart_xsum', 't5', 'bart_base', 'pegasus_xsum', 'bart_cnn', 'original'],
    ['pegasus_xsum', 'bart_xsum', 'bart_cnn', 'bart_base', 'original', 'gpt2'],
    ['original', 'pegasus_xsum', 'gpt2', 'bart_base', 'bart_cnn', 't5'],
    ['original', 'pegasus_xsum', 'bart_base', 'bart_xsum', 'gpt2', 't5'],
    ['bart_xsum', 'bart_cnn', 'gpt2', 'bart_base', 't5', 'original'],
    ['bart_xsum', 't5', 'original', 'gpt2', 'bart_base', 'pegasus_xsum'],
    ['bart_cnn', 'gpt2', 't5', 'original', 'pegasus_xsum', 'bart_base'],
    ['t5', 'bart_cnn', 'bart_base', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['bart_cnn', 't5', 'gpt2', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['original', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum', 'bart_base'],
    ['original', 'gpt2', 't5', 'bart_xsum', 'bart_base', 'bart_cnn'],
    ['t5', 'bart_base', 'pegasus_xsum', 'gpt2', 'bart_cnn', 'original'],
    ['gpt2', 'bart_xsum', 'bart_base', 'original', 'bart_cnn', 'pegasus_xsum'],
    ['bart_base', 'gpt2', 'bart_cnn', 't5', 'bart_xsum', 'original'],
    ['bart_cnn', 'bart_base', 'original', 'bart_xsum', 't5', 'gpt2'],
    ['bart_cnn', 'original', 't5', 'gpt2', 'pegasus_xsum', 'bart_xsum'],
    ['bart_cnn', 'gpt2', 't5', 'bart_base', 'pegasus_xsum', 'original'],
    ['original', 'gpt2', 'bart_cnn', 'bart_base', 'pegasus_xsum', 't5'],
    ['original', 'gpt2', 'bart_cnn', 't5', 'bart_xsum', 'bart_base'],
    ['gpt2', 'bart_cnn', 'pegasus_xsum', 'bart_base', 'original', 'bart_xsum'],
    ['original', 'bart_cnn', 'bart_xsum', 'pegasus_xsum', 't5', 'gpt2'],
    ['t5', 'gpt2', 'pegasus_xsum', 'bart_cnn', 'bart_xsum', 'original'],
    ['pegasus_xsum', 'bart_cnn', 'gpt2', 't5', 'bart_base', 'original'],
    ['bart_cnn', 'pegasus_xsum', 'original', 'bart_base', 'bart_xsum', 'gpt2'],
    ['original', 'bart_base', 't5', 'gpt2', 'pegasus_xsum', 'bart_cnn'],
    ['gpt2', 'bart_base', 't5', 'pegasus_xsum', 'original', 'bart_xsum'],
    ['pegasus_xsum', 'original', 'gpt2', 't5', 'bart_cnn', 'bart_base'],
    ['pegasus_xsum', 't5', 'bart_base', 'original', 'bart_xsum', 'bart_cnn'],
    ['gpt2', 'original', 'bart_cnn', 'bart_xsum', 't5', 'pegasus_xsum'],
    ['bart_cnn', 'bart_xsum', 'original', 'bart_base', 'pegasus_xsum', 'gpt2']]

def map2index(ls):
  r = []
  for i in ls:
    if i == "original":
      r.append(0)
    if i == "bart_base":
      r.append(1)
    if i == "bart_cnn":
      r.append(2)
    if i == "bart_xsum":
      r.append(3)
    if i == "t5":
      r.append(4)
    if i == "gpt2":
      r.append(5)
    if i == "pegasus_xsum":
      r.append(6)
  return r