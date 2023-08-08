# # Merge Two dataset
# import pandas as pd
# colnames = ['user_id', 'trajectory', 'timestamp', 'category']
# colnames1 = ['user_id', 'trajectory', 'timestamp', 'category','altitude']
# df = pd.read_csv("data/mainTUL_Geolife.csv", names=colnames, header=None)
# df1 = pd.read_csv("data/mainTUL_Geolife_more_than_700_car.csv", names=colnames1, header=None)
# df1.drop('altitude', axis=1, inplace=True)
# concat_df = pd.concat([df, df1], ignore_index=True)
# print(concat_df['user_id'])
# concat_df.to_csv('data/GeoLife_merged.csv', index=False)

import random
import pandas as pd
import math

# Load your original dataset into a pandas DataFrame
# colnames = ['user_id', 'trajectory', 'timestamp', 'category']
# df = pd.read_csv("data/GeoLife_merged.csv")
df = pd.read_csv("Data/staticmap.csv")
# df = pd.read_csv("Data/osmnx.csv")
print(df)

# Get a list of all unique user IDs
user_ids = df["user_id"].unique().tolist()
print(user_ids)
# print(len(user_ids))
train_data = []
test_data = []
for user_id in user_ids:
    # Extract all the input data points for this user ID
    user_data = df[df["user_id"] == user_id]
    if len(user_data) > 2:
        if len(user_data) >= 8:
            train_size = math.floor(len(user_data) * .8)
            train_user_data = user_data[:train_size]
            test_user_data = user_data[train_size:]

        elif len(user_data) > 5:
            # Split the input data points into training and test sets
            train_user_data = user_data[:(len(user_data) - 2)]
            test_user_data = user_data[(len(user_data) - 2):]
        elif len(user_data) > 2:
            train_user_data = user_data[:(len(user_data) - 1)]
            test_user_data = user_data[(len(user_data) - 1):]
        # Append the training data to the train_data list
        train_data.append(train_user_data)
        # Append the test data to the test_data list
        test_data.append(test_user_data)

# Concatenate the training and test data into single DataFrames
train_data = pd.concat(train_data)
test_data = pd.concat(test_data)
# print(train_data)
# print(test_data)

# # Save the training and test sets to separate CSV files
# train_data.to_csv("data/merged_train_dataset.csv", index=False)
# test_data.to_csv("data/merged_test_dataset.csv", index=False)

# Save the training and test sets to separate CSV files
# train_data.to_csv("data/firstpresentation/merged_train_dataset.csv", index=False)
print(test_data)
print(len(df["user_id"].unique().tolist()))
test_data.to_csv("Data/small_static.csv", index=False)
