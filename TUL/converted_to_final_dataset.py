import pandas as pd
import ast
from collections import Counter


def parse_list_col(value):
    return ast.literal_eval(value)


# df = pd.read_csv('data/newData/merged_updated_test.csv')
# df_train=pd.read_csv('data/newData/merged_updated_train.csv')

df = pd.read_csv('data/firstpresentation/separate/merged_updated_test.csv')
df_train = pd.read_csv('data/firstpresentation/separate/merged_updated_train.csv')
osmnx_df = pd.read_csv('data/firstpresentation/separate/merged_updated_osmnx.csv')
staticmap_df = pd.read_csv('data/firstpresentation/separate/merged_updated_staticmap.csv')
secure_osmnx_df = pd.read_csv('data/firstpresentation/merged_updated_secure_osmnx.csv')
secure_staticmap_df = pd.read_csv('data/firstpresentation/merged_updated_secure_staticmap.csv')
kosmnx_df = pd.read_csv('data/firstpresentation/merged_updated_kosmnx.csv')
kstaticmap_df = pd.read_csv('data/firstpresentation/merged_updated_kstaticmap.csv')

df.rename(columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
          inplace=True)
df_train.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
osmnx_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
staticmap_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)

secure_osmnx_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
secure_staticmap_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
kosmnx_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
kstaticmap_df.rename(
    columns={'user_encoded': 'user_id', 'encoded_location_id': 'location_id', 'encoded_timestamp': 'timestamp'},
    inplace=True)
max_lat=0
max_lon=0

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

# def conversion(df):
#     for i in range(len(df_train['location_id'])):
#         list = [int(x) for x in df_train['location_id'][i].strip('[]').split()]
#         time = [int(x) for x in df_train['timestamp'][i].strip('[]').split()]
#         df_train['location_id'][i] = list
#         df_train['timestamp'][i] = time


def conversion(df):
    lenList = []
    lenTime = []
    for i in range(len(df['location_id'])):
        list = []
        time = []

        for x in df['location_id'][i].strip('[]').split():
            if x != "...":
                list.append(int(x))
        for x in df['timestamp'][i].strip('[]').split():
            if x != "...":
                time.append(int(x))

        if len(list)>300 or len(time)>300:
            df['location_id'][i] = list[:300]
            df['timestamp'][i] = time[:300]
        else:
            df['location_id'][i] = list
            df['timestamp'][i] = time

    #     lenList.append(len(df['location_id'][i]))
    #     lenTime.append(len(df['timestamp'][i]))
    #     # if len(list) > max_lat:
    #     #     max_lat = len(list)
    #     # if len(time) > max_lon:
    #     #     max_lon = len(time)
    # print("*********")
    # # Count the occurrences of each unique value
    # print(lenList)
    # counted_list = Counter(lenList)
    # counted_time = Counter(lenTime)
    # # Sort the values in descending order based on count
    # sorted_list = sorted(counted_list.items(), key=lambda x: x[0], reverse=False)
    #
    # # Sort the values in descending order based on count
    # sorted_time = sorted(counted_time.items(), key=lambda x: x[0], reverse=False)
    # # Print the count of each unique value
    # for value, count in sorted_list:
    #     print(f"{value}: {count}")
    # print("$$$$$$$$$$$$$$$$$$$$$")
    # for value, count in sorted_time:
    #     print(f"{value}: {count}")
    return df


osmnx_df = conversion(osmnx_df)
staticmap_df = conversion(staticmap_df)
secure_osmnx_df = conversion(secure_osmnx_df)
secure_staticmap_df = conversion(secure_staticmap_df)
kosmnx_df = conversion(kosmnx_df)
kstaticmap_df = conversion(kstaticmap_df)




# print(df_evaluation)
df_train.to_csv('data/firstpresentation/merged_final_train.csv', index=False)
df.to_csv('data/firstpresentation/merged_final_test.csv', index=False)
osmnx_df.to_csv('data/firstpresentation/merged_final_osmnx.csv', index=False)
staticmap_df.to_csv('data/firstpresentation/merged_final_staticmap.csv', index=False)
secure_osmnx_df.to_csv('data/firstpresentation/merged_final_secure_osmnx.csv', index=False)
secure_staticmap_df.to_csv('data/firstpresentation/merged_final_secure_staticmap.csv', index=False)
kosmnx_df.to_csv('data/firstpresentation/merged_final_kosmnx.csv', index=False)
kstaticmap_df.to_csv('data/firstpresentation/merged_final_kstaticmap.csv', index=False)



# for i in range(len(df_privacy['location_id'])):
#     list=[]
#     time=[]
#     for x in df_privacy['location_id'][i].strip('[]').split():
#         if x!= "...":
#             list.append(int(x))
#     for x in df_privacy['timestamp'][i].strip('[]').split():
#         if x != "...":
#             time.append(int(x))
#     df_privacy['location_id'][i] = list
#     df_privacy['timestamp'][i] = time
# for i in range(len(df_privacy['location_id'])):
#     list = [x for x in df_privacy['location_id'][i].strip('[]').split()]
#     time = [x for x in df_privacy['timestamp'][i].strip('[]').split()]
#     df_privacy['location_id'][i] = list
#     df_privacy['timestamp'][i] = time
## Testing
# df_train=df_train[:1]
# for i in range(len(df_train['location_id'])):
#     for x in df_train['location_id'][i].strip('[]').split():
#         print(type(x))

# df_evaluation = df_evaluation[:10]
# for i in range(len(df_evaluation['location_id'])):
#     for x in df_evaluation['location_id'][i].strip('[]').split():
#         print(type(x))
#         print(x)
#         print(int(x))


# list = [int(x) for x in df_train['location_id'][i].strip('[]').split()]
# time = [int(x) for x in df_train['timestamp'][i].strip('[]').split()]
# df_train['location_id'][i] = list
# df_train['timestamp'][i] = time
