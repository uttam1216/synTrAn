import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import numpy as np


# define a custom function to parse the list column
def parse_list_col(value):
    return ast.literal_eval(value)


# load the train and test csv files
train_df = pd.read_csv('data/merged_train_dataset.csv',
                       converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
test_df = pd.read_csv('data/merged_test_dataset.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})

# concatenate the train and test datasets for consistency in encoding
combined_df = pd.concat([train_df, test_df], ignore_index=True)
# print(combined_df['location_id'])
# extract the column with nested list
col_name = 'location_id'
col_name1 = 'timestamp'
col_name2 = 'user_id'
col_name3 = 'category'

# # flatten the nested list column
flat_list = np.concatenate(combined_df[col_name]).ravel().tolist()
flat_list1 = np.concatenate(combined_df[col_name1]).ravel().tolist()

# initialize the LabelEncoder
le = LabelEncoder()
le_timestamp = LabelEncoder()
le_user = LabelEncoder()

# fit the LabelEncoder to the flattened list
# le_user.fit(combined_df[col_name2])
le.fit(flat_list)
le_timestamp.fit(flat_list1)

# Fit and transform the column in the train dataframe
train_df['user_encoded'] = le_user.fit_transform(train_df['user_id'])

# Transform the column in the test dataframe using the fitted encoder
test_df['user_encoded'] = le_user.transform(test_df['user_id'])

# print(flat_list)

# transform the nested list column in train and test datasets using the fitted LabelEncoder
# train_df['encoded_'+col_name2] = train_df[col_name2].apply(lambda x: le_user.transform(x))
# test_df['encoded_'+col_name2] = test_df[col_name2].apply(lambda x: le_user.transform(x))

train_df['encoded_' + col_name] = train_df[col_name].apply(lambda x: le.transform(x))
test_df['encoded_' + col_name] = test_df[col_name].apply(lambda x: le.transform(x))

train_df['encoded_' + col_name1] = train_df[col_name1].apply(lambda x: le_timestamp.transform(x))
test_df['encoded_' + col_name1] = test_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# drop the original nested list column from both train and test datasets
train_df.drop(col_name, axis=1, inplace=True)
test_df.drop(col_name, axis=1, inplace=True)
train_df.drop(col_name2, axis=1, inplace=True)
test_df.drop(col_name2, axis=1, inplace=True)

train_df.drop(col_name1, axis=1, inplace=True)
test_df.drop(col_name1, axis=1, inplace=True)
train_df.drop(col_name3, axis=1, inplace=True)
test_df.drop(col_name3, axis=1, inplace=True)

# save the updated train and test csv files
train_df.to_csv('data/merged_updated_train.csv', index=False)
test_df.to_csv('data/merged_updated_test.csv', index=False)
