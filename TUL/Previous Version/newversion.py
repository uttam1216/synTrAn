import pandas as pd
import ast


def parse_list_col(value):
    return ast.literal_eval(value)


df = pd.read_csv('data/merged_updated_test.csv')
df_train=pd.read_csv('data/merged_updated_train.csv')
print(type(df['location_id']))

for i in range(len(df['location_id'])):
    list = [int(x) for x in df['location_id'][i].strip('[]').split()]
    time = [int(x) for x in df['timestamp'][i].strip('[]').split()]
    df['location_id'][i] = list
    df['timestamp'][i] = time
for i in range(len(df_train['location_id'])):
    list = [int(x) for x in df_train['location_id'][i].strip('[]').split()]
    time = [int(x) for x in df_train['timestamp'][i].strip('[]').split()]
    df_train['location_id'][i] = list
    df_train['timestamp'][i] = time


num_classes=df_train['user_id'].nunique()
print(num_classes)
# save the updated train and test csv files
df_train.to_csv('data/merged_final_train.csv', index=False)
df.to_csv('data/merged_final_test.csv', index=False)
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from collections import defaultdict
#
# from collections import defaultdict
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
#
# from collections import defaultdict
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
#
#
# class ListLabelEncoder:
#     def __init__(self):
#         self.label_encoders = defaultdict(LabelEncoder)
#
#     def fit_transform(self, data):
#         transformed_data = []
#         for row in data:
#             transformed_row = []
#             for i in range(len(row)):
#                 if isinstance(row[i], list):
#                     transformed = self.label_encoders[i].fit_transform(row[i])
#                 else:
#                     transformed = self.label_encoders[i].fit_transform([row[i]])
#                     transformed = transformed[0]
#                 transformed_row.append(transformed)
#             transformed_data.append(transformed_row)
#         return np.array(transformed_data)
#
#     def transform(self, data):
#         transformed_data = []
#         for row in data:
#             transformed_row = []
#             for i in range(len(row)):
#                 if isinstance(row[i], list):
#                     transformed = self.label_encoders[i].transform(row[i])
#                 else:
#                     transformed = self.label_encoders[i].transform([row[i]])
#                     transformed = transformed[0]
#                 transformed_row.append(transformed)
#             transformed_data.append(transformed_row)
#         return np.array(transformed_data)
#
#
# train_df = pd.read_csv("data/train_dataset.csv")
# test_df = pd.read_csv("data/test_dataset.csv")
# # value = test_df.isnull().values.any()
# # print(value)
#
# le_loc = ListLabelEncoder()
# train_df['location_id'] = le_loc.fit_transform(train_df['location_id'].tolist())
# test_df['location_id'] = le_loc.transform(test_df['location_id'].tolist())
#
# # fit and transform location IDs in the combined dataset
# # combined_locs_encoded = le_loc.fit_transform(train_df['location_id'].tolist() + test_df['location_id'].tolist())
# #
# # # transform location IDs in the train and test datasets
# # train_locs_encoded = combined_locs_encoded[:len(train_df)]
# # test_locs_encoded = combined_locs_encoded[len(train_df):]
# # print(train_locs_encoded)
#
# le_user = LabelEncoder()
# train_df['user_id'] = le_user.fit_transform(train_df['user_id'].tolist())
# test_df['user_id'] = le_user.transform(test_df['user_id'].tolist())
#
# le_timestamp = ListLabelEncoder()
# train_df['timestamp'] = le_timestamp.fit_transform(train_df['timestamp'].tolist())
# test_df['timestamp'] = le_timestamp.transform(test_df['timestamp'].tolist())
