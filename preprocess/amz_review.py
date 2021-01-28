from transformers import BertModel, BertTokenizer
import torch 
import os
import numpy as np
import pandas as pd 

# create model
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
model = BertModel.from_pretrained('bert-large-cased')

# read data
root = '/home/alan/Downloads/imbalanced/amz_review'
labels = []

# minority data
cat0_num = 738943
index = np.arange(cat0_num)
np.random.shuffle(index)
index = set(index.tolist()[:int(cat0_num * 0.002)])

cat0 = []
with open(os.path.join(root, 'amazon_books_cat0.txt')) as f:
    i = 0
    for line in f:
        print("line: {}/{} {:.2f}%".format(i, cat0_num, i/cat0_num*100), end='\r')
        if i in index:
            tokens = tokenizer.encode(line, add_special_tokens=True, max_length=512)
            input_ids = torch.tensor([tokens])
            with torch.no_grad():
                last_hidden_state = model(input_ids)[0]

            embedding = last_hidden_state[:, 0, :]
            cat0.append(embedding)
        i += 1
cat0 = torch.cat(cat0, dim=0)
labels += [1] * len(cat0)
print()
print(cat0.shape, len(labels))

# majority class
cat1_num = 7203909
index = np.arange(cat1_num)
np.random.shuffle(index)
index = set(index.tolist()[:int(cat1_num * 0.002)])

cat1 = []
with open(os.path.join(root, 'amazon_books_cat1.txt')) as f:
    i = 0
    for line in f:
        print("line: {}/{} {:.2f}%".format(i, cat1_num, i/cat1_num*100), end='\r')
        if i in index:
            tokens = tokenizer.encode(line, add_special_tokens=True, max_length=512)
            input_ids = torch.tensor([tokens])
            with torch.no_grad():
                last_hidden_state = model(input_ids)[0]

            embedding = last_hidden_state[:, 0, :]
            cat1.append(embedding)
        i += 1
cat1 = torch.cat(cat1, dim=0)
labels += [0] * len(cat1)
print()
print(cat1.shape, len(labels)-len(cat0))

x = torch.cat((cat0, cat1), dim=0)
labels = torch.from_numpy(np.array(labels).reshape(-1, 1))
print(x.shape, labels.shape)

data = torch.cat((x, labels), dim=1).numpy()
print(data.shape)

# save
pd.DataFrame(data).to_csv(os.path.join(root, 'books2.csv'), header=None, index=False)

# train valid test split
x, labels = data[:, :-1], data[:, -1]
train_ratio = 0.6
valid_ratio = 0.1
index = np.arange(len(data))
np.random.shuffle(index)
train_index = index[:int(len(index)*train_ratio)]
valid_index = index[int(len(index)*train_ratio):int(len(index)*(train_ratio+valid_ratio))]
test_index = index[int(len(index)*(train_ratio+valid_ratio)):]

# print class distribution
from collections import Counter
print(Counter(labels[train_index]))
print(Counter(labels[valid_index]))
print(Counter(labels[test_index]))
pd.DataFrame(train_index).to_csv(os.path.join(root, 'books2_{}_train_idx.csv'.format(train_ratio)), header=None, index=False)
pd.DataFrame(valid_index).to_csv(os.path.join(root, 'books2_{}_valid_idx.csv'.format(train_ratio)), header=None, index=False)
pd.DataFrame(test_index).to_csv(os.path.join(root, 'books2_{}_test_idx.csv'.format(train_ratio)), header=None, index=False)