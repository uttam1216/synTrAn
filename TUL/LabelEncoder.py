# import pandas as pd
# import ast
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
#
#
# # define a custom function to parse the list column
# def parse_list_col(value):
#     return ast.literal_eval(value)
#
#
# # # load the train and test csv files train_df = pd.read_csv('data/merged_train_dataset.csv', converters={
# # 'location_id': parse_list_col, 'timestamp': parse_list_col}) test_df = pd.read_csv('data/merged_test_dataset.csv',
# # converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#
# # load the train and test csv files
# train_df = pd.read_csv('data/StaticMap/merged_train_dataset.csv',
#                        converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# test_df = pd.read_csv('data/StaticMap/merged_test_dataset.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#
#
# # evaluation_df = pd.read_csv('data/StaticMap/staticmap_10.csv', converters={'location_id': parse_list_col,
# # 'timestamp': parse_list_col}) privacy_preserve_df = pd.read_csv('data/StaticMap/temp_30_staticmap.csv',
# # converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#
#
# # evaluation_df = pd.read_csv('data/k_anonymity/staticmap_10.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# evaluation_df = pd.read_csv('data/k_anonymity/osmnx_10.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#
# # privacy_preserve_df = pd.read_csv('data/k_anonymity/final_k_anonymity_clustering.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# # privacy_preserve_df = pd.read_csv('data/k_anonymity/temp_370_staticmap.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# privacy_preserve_df = pd.read_csv('data/k_anonymity/final_k_anonymity_clustering_osmnx.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
#
#
# # evaluation_df = pd.read_csv('data/Final_data/staticmap.csv', converters={'location_id': parse_list_col,
# # 'timestamp': parse_list_col}) privacy_preserve_df = pd.read_csv('data/Final_data/osmnx.csv', converters={
# # 'location_id': parse_list_col, 'timestamp': parse_list_col})
#
#
# # train_df.rename(columns={'user_id': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
# #           inplace=True)
# # test_df.rename(
# #     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
# #     inplace=True)
# # evaluation_df.rename(
# #     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
# #     inplace=True)
# # privacy_preserve_df.rename(
# #     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
# #     inplace=True)
#
# # concatenate the train and test datasets for consistency in encoding
# combined_df = pd.concat([train_df, test_df,evaluation_df,privacy_preserve_df], ignore_index=True)
# # combined_df = pd.concat([train_df, test_df], ignore_index=True)
# print(combined_df.columns)
# # print(combined_df['location_id'])
# # extract the column with nested list
# col_name = 'location_id'
# col_name1 = 'timestamp'
# col_name2 = 'user_id'
# # col_name3 = 'category'
#
# # # flatten the nested list column
# flat_list = np.concatenate(combined_df[col_name]).ravel().tolist()
# flat_list1 = np.concatenate(combined_df[col_name1]).ravel().tolist()
#
# # initialize the LabelEncoder
# le = LabelEncoder()
# le_timestamp = LabelEncoder()
# le_user = LabelEncoder()
#
# # fit the LabelEncoder to the flattened list
# # le_user.fit(combined_df[col_name2])
# le.fit(flat_list)
# le_timestamp.fit(flat_list1)
#
# # Fit and transform the column in the train dataframe
# train_df['user_encoded'] = le_user.fit_transform(train_df['user_id'])
#
# # Transform the column in the test dataframe using the fitted encoder
# test_df['user_encoded'] = le_user.transform(test_df['user_id'])
#
# # Transform the column in the test dataframe using the fitted encoder
# evaluation_df['user_encoded'] = le_user.transform(evaluation_df['user_id'])
# privacy_preserve_df['user_encoded'] = le_user.transform(privacy_preserve_df['user_id'])
#
# # print(flat_list)
#
# # transform the nested list column in train and test datasets using the fitted LabelEncoder
# # train_df['encoded_'+col_name2] = train_df[col_name2].apply(lambda x: le_user.transform(x))
# # test_df['encoded_'+col_name2] = test_df[col_name2].apply(lambda x: le_user.transform(x))
#
# train_df['encoded_' + col_name] = train_df[col_name].apply(lambda x: le.transform(x))
# test_df['encoded_' + col_name] = test_df[col_name].apply(lambda x: le.transform(x))
# evaluation_df['encoded_' + col_name] = evaluation_df[col_name].apply(lambda x: le.transform(x))
# privacy_preserve_df['encoded_' + col_name] = privacy_preserve_df[col_name].apply(lambda x: le.transform(x))
#
# train_df['encoded_' + col_name1] = train_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# test_df['encoded_' + col_name1] = test_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# evaluation_df['encoded_' + col_name1] = evaluation_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# privacy_preserve_df['encoded_' + col_name1] = privacy_preserve_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# # drop the original nested list column from both train and test datasets
# train_df.drop(col_name, axis=1, inplace=True)
# test_df.drop(col_name, axis=1, inplace=True)
# evaluation_df.drop(col_name, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name, axis=1, inplace=True)
# train_df.drop(col_name2, axis=1, inplace=True)
# test_df.drop(col_name2, axis=1, inplace=True)
# evaluation_df.drop(col_name2, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name2, axis=1, inplace=True)
#
# train_df.drop(col_name1, axis=1, inplace=True)
# test_df.drop(col_name1, axis=1, inplace=True)
# evaluation_df.drop(col_name1, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name1, axis=1, inplace=True)
# # train_df.drop(col_name3, axis=1, inplace=True)
# # test_df.drop(col_name3, axis=1, inplace=True)
#
# # save the updated train and test csv files
# train_df.to_csv('data/StaticMap/merged_updated_train.csv', index=False)
# test_df.to_csv('data/StaticMap/merged_updated_test.csv', index=False)
# # evaluation_df.to_csv('data/StaticMap/merged_updated_evaluation.csv', index=False)
# # privacy_preserve_df.to_csv('data/StaticMap/merged_updated_privacy.csv', index=False)
#
# evaluation_df.to_csv('data/k_anonymity/merged_updated_evaluation.csv', index=False)
# privacy_preserve_df.to_csv('data/k_anonymity/merged_updated_privacy.csv', index=False)
#


# Testing For better accuracy

import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import numpy as np


# define a custom function to parse the list column
def parse_list_col(value):
    return ast.literal_eval(value)


# load the train and test csv files
train_df = pd.read_csv('data/firstpresentation/merged_train_dataset.csv',
                       converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
test_df = pd.read_csv('data/firstpresentation/merged_test_dataset.csv',
                      converters={'location_id': parse_list_col, 'timestamp': parse_list_col})

# Synthetic Dataset
osmnx_df = pd.read_csv('data/firstpresentation/osmnx_10.csv',
                       converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
staticmap_df = pd.read_csv('data/firstpresentation/staticmap_10.csv',
                           converters={'location_id': parse_list_col, 'timestamp': parse_list_col})

# Privacy Preserve Dataset
# Secure Start
# secure_osmnx_df = pd.read_csv('data/firstpresentation/secure_osmnx.csv',
#                               converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
secure_staticmap_df = pd.read_csv('data/firstpresentation/resultcsvV1.csv',
                                  converters={'location_id': parse_list_col, 'timestamp': parse_list_col})

# K-anonymity
# kosmnx_df = pd.read_csv('data/firstpresentation/kosmnx.csv',
#                         converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# kstaticmap_df = pd.read_csv('data/firstpresentation/kstaticmap.csv',
#                             converters={'location_id': parse_list_col, 'timestamp': parse_list_col})

# concatenate the train and test datasets for consistency in encoding
# combined_df = pd.concat([train_df, test_df,evaluation_df,privacy_preserve_df], ignore_index=True)
# combined_df = pd.concat(
#     [train_df, test_df, osmnx_df, staticmap_df, secure_osmnx_df, secure_staticmap_df, kosmnx_df, kstaticmap_df],
#     ignore_index=True)

combined_df = pd.concat(
    [train_df, test_df, osmnx_df, staticmap_df, secure_staticmap_df],
    ignore_index=True)
print(combined_df.columns)
# print(combined_df['location_id'])
# extract the column with nested list
col_name = 'location_id'
col_name1 = 'timestamp'
col_name2 = 'user_id'
# col_name3 = 'category'

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

osmnx_df['user_encoded'] = le_user.transform(osmnx_df['user_id'])
staticmap_df['user_encoded'] = le_user.transform(staticmap_df['user_id'])
# secure_osmnx_df['user_encoded'] = le_user.transform(secure_osmnx_df['user_id'])
secure_staticmap_df['user_encoded'] = le_user.transform(secure_staticmap_df['user_id'])
# kosmnx_df['user_encoded'] = le_user.transform(kosmnx_df['user_id'])
# kstaticmap_df['user_encoded'] = le_user.transform(kstaticmap_df['user_id'])

# Transform the column in the test dataframe using the fitted encoder
# evaluation_df['user_encoded'] = le_user.transform(evaluation_df['user_id'])
# privacy_preserve_df['user_encoded'] = le_user.transform(privacy_preserve_df['user_id'])

# print(flat_list)

# transform the nested list column in train and test datasets using the fitted LabelEncoder
# train_df['encoded_'+col_name2] = train_df[col_name2].apply(lambda x: le_user.transform(x))
# test_df['encoded_'+col_name2] = test_df[col_name2].apply(lambda x: le_user.transform(x))

train_df['encoded_' + col_name] = train_df[col_name].apply(lambda x: le.transform(x))
test_df['encoded_' + col_name] = test_df[col_name].apply(lambda x: le.transform(x))
osmnx_df['encoded_' + col_name] = osmnx_df[col_name].apply(lambda x: le.transform(x))
staticmap_df['encoded_' + col_name] = staticmap_df[col_name].apply(lambda x: le.transform(x))
# secure_osmnx_df['encoded_' + col_name] = secure_osmnx_df[col_name].apply(lambda x: le.transform(x))
secure_staticmap_df['encoded_' + col_name] = secure_staticmap_df[col_name].apply(lambda x: le.transform(x))
# kosmnx_df['encoded_' + col_name] = kosmnx_df[col_name].apply(lambda x: le.transform(x))
# kstaticmap_df['encoded_' + col_name] = kstaticmap_df[col_name].apply(lambda x: le.transform(x))
# evaluation_df['encoded_' + col_name] = evaluation_df[col_name].apply(lambda x: le.transform(x))
# privacy_preserve_df['encoded_' + col_name] = privacy_preserve_df[col_name].apply(lambda x: le.transform(x))

train_df['encoded_' + col_name1] = train_df[col_name1].apply(lambda x: le_timestamp.transform(x))
test_df['encoded_' + col_name1] = test_df[col_name1].apply(lambda x: le_timestamp.transform(x))
osmnx_df['encoded_' + col_name1] = osmnx_df[col_name1].apply(lambda x: le_timestamp.transform(x))
staticmap_df['encoded_' + col_name1] = staticmap_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# secure_osmnx_df['encoded_' + col_name1] = secure_osmnx_df[col_name1].apply(lambda x: le_timestamp.transform(x))
secure_staticmap_df['encoded_' + col_name1] = secure_staticmap_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# kosmnx_df['encoded_' + col_name1] = kosmnx_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# kstaticmap_df['encoded_' + col_name1] = kstaticmap_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# evaluation_df['encoded_' + col_name1] = evaluation_df[col_name1].apply(lambda x: le_timestamp.transform(x))
# privacy_preserve_df['encoded_' + col_name1] = privacy_preserve_df[col_name1].apply(lambda x:
# le_timestamp.transform(x)) drop the original nested list column from both train and test datasets
train_df.drop(col_name, axis=1, inplace=True)
test_df.drop(col_name, axis=1, inplace=True)
osmnx_df.drop(col_name, axis=1, inplace=True)
staticmap_df.drop(col_name, axis=1, inplace=True)
# secure_osmnx_df.drop(col_name, axis=1, inplace=True)
secure_staticmap_df.drop(col_name, axis=1, inplace=True)
# kosmnx_df.drop(col_name, axis=1, inplace=True)
# kstaticmap_df.drop(col_name, axis=1, inplace=True)
# evaluation_df.drop(col_name, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name, axis=1, inplace=True)
train_df.drop(col_name2, axis=1, inplace=True)
test_df.drop(col_name2, axis=1, inplace=True)
osmnx_df.drop(col_name2, axis=1, inplace=True)
staticmap_df.drop(col_name2, axis=1, inplace=True)
# secure_osmnx_df.drop(col_name2, axis=1, inplace=True)
secure_staticmap_df.drop(col_name2, axis=1, inplace=True)
# kosmnx_df.drop(col_name2, axis=1, inplace=True)
# kstaticmap_df.drop(col_name2, axis=1, inplace=True)
# evaluation_df.drop(col_name2, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name2, axis=1, inplace=True)

train_df.drop(col_name1, axis=1, inplace=True)
test_df.drop(col_name1, axis=1, inplace=True)
osmnx_df.drop(col_name1, axis=1, inplace=True)
staticmap_df.drop(col_name1, axis=1, inplace=True)
# secure_osmnx_df.drop(col_name1, axis=1, inplace=True)
secure_staticmap_df.drop(col_name1, axis=1, inplace=True)
# kosmnx_df.drop(col_name1, axis=1, inplace=True)
# kstaticmap_df.drop(col_name1, axis=1, inplace=True)
# evaluation_df.drop(col_name1, axis=1, inplace=True)
# privacy_preserve_df.drop(col_name1, axis=1, inplace=True)
# train_df.drop(col_name3, axis=1, inplace=True)
# test_df.drop(col_name3, axis=1, inplace=True)

# save the updated train and test csv files
train_df.to_csv('data/firstpresentation/merged_updated_train.csv', index=False)
test_df.to_csv('data/firstpresentation/merged_updated_test.csv', index=False)
osmnx_df.to_csv('data/firstpresentation/merged_updated_osmnx.csv', index=False)
staticmap_df.to_csv('data/firstpresentation/merged_updated_staticmap.csv', index=False)
# secure_osmnx_df.to_csv('data/firstpresentation/merged_updated_secure_osmnx.csv', index=False)
secure_staticmap_df.to_csv('data/firstpresentation/merged_updated_secure_staticmap.csv', index=False)
# kosmnx_df.to_csv('data/firstpresentation/merged_updated_kosmnx.csv', index=False)
# kstaticmap_df.to_csv('data/firstpresentation/merged_updated_kstaticmap.csv', index=False)
# evaluation_df.to_csv('data/StaticMap/merged_updated_evaluation.csv', index=False)
# privacy_preserve_df.to_csv('data/StaticMap/merged_updated_privacy.csv', index=False)

# evaluation_df.to_csv('data/k_anonymity/merged_updated_evaluation.csv', index=False)
# privacy_preserve_df.to_csv('data/k_anonymity/merged_updated_privacy.csv', index=False)


# # evaluation_df = pd.read_csv('data/k_anonymity/staticmap_10.csv', converters={'location_id': parse_list_col,
# 'timestamp': parse_list_col}) evaluation_df = pd.read_csv('data/k_anonymity/osmnx_10.csv', converters={
# 'location_id': parse_list_col, 'timestamp': parse_list_col})
#
# # privacy_preserve_df = pd.read_csv('data/k_anonymity/final_k_anonymity_clustering.csv', converters={'location_id':
# parse_list_col, 'timestamp': parse_list_col}) # privacy_preserve_df = pd.read_csv(
# 'data/k_anonymity/temp_370_staticmap.csv', converters={'location_id': parse_list_col, 'timestamp': parse_list_col})
# privacy_preserve_df = pd.read_csv('data/k_anonymity/final_k_anonymity_clustering_osmnx.csv', converters={
# 'location_id': parse_list_col, 'timestamp': parse_list_col})


# evaluation_df = pd.read_csv('data/Final_data/staticmap.csv', converters={'location_id': parse_list_col,
# 'timestamp': parse_list_col}) privacy_preserve_df = pd.read_csv('data/Final_data/osmnx.csv', converters={
# 'location_id': parse_list_col, 'timestamp': parse_list_col})


# train_df.rename(columns={'user_id': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
#           inplace=True)
# test_df.rename(
#     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
#     inplace=True)
# evaluation_df.rename(
#     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
#     inplace=True)
# privacy_preserve_df.rename(
#     columns={'user_encoded': 'user_id', 'latitude': 'location_id', 'longitude': 'timestamp'},
#     inplace=True)
