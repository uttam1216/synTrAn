import pandas as pd
df=pd.read_csv('data/train_dataset.csv')
# # print(df.dtypes)
# # print(type(df['timestamp'].values.reshape(-1,1)))
#
# print(type(df.iloc[0]['location_id']))
# print(df.iloc[0]['location_id'])
# a=df.iloc[0]['location_id'].split()
# print(a)


from sklearn.preprocessing import OneHotEncoder
import numpy as np

# create some sample data
# data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# data=np.array(df['timestamp'])
# print(data)
# data=data.reshape(-1,1)
# print(data)
# # create the OneHotEncoder object
# encoder = OneHotEncoder()
#
#
# # fit the encoder to the data
# encoder.fit(data)
#
# # transform the data using the encoder
# onehot_data = encoder.transform(data).toarray()
#
# # print the one-hot encoded data
# print(onehot_data)
import torch.nn as nn
import torch
import ast

unique_values = set()
for row in df['timestamp']:
    # print(type(row))
    for value in row[1:-1].split(', '):
        # print(value)
        unique_values.add(value)

vocab_size = len(unique_values)
print(vocab_size)
embedding_dim = 32
embedding = nn.Embedding(vocab_size, embedding_dim)

time = df['timestamp'].apply(ast.literal_eval).tolist()  # Convert string to list of lists
time_tensors = [torch.LongTensor(seq) for seq in time]
# Pass the tensor through the Embedding layer
embedded = embedding(time_tensors)
print(embedded)
# time=[]
# for row in df['timestamp']:
#     temp=ast.literal_eval(row)
#     time.append(temp)

# time=ast.literal_eval(df['timestamp'])
# seq_tensor = torch.LongTensor(torch.LongTensor(seq) for seq in time)
