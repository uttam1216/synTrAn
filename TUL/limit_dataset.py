import pandas as pd
# pd.read_csv('data/firstpresentation/resultcsv.csv'
# ground_df=pd.read_csv("data/newData/TUL_dataset.csv")
# synthetic_df=pd.read_csv("data/Final_data/osmnx.csv")
# privacy_df=pd.read_csv("data/Final_data/staticmap.csv")
# print(ground_df['user_id'].unique())
# print(synthetic_df['user_id'].unique())
# print(privacy_df['user_id'].unique())
#
# edit_synthetic_df=synthetic_df.loc[synthetic_df['user_id'].isin([10,20,52,56,58,62,64,65,67])]
# edit_privacy_df=privacy_df.loc[privacy_df['user_id'].isin([10,20,52,56,58,62,64,65,67])]
# print(edit_synthetic_df)
# print(edit_privacy_df)
#
# edit_synthetic_df.to_csv("data/Final_data/Synthetic_trajectories.csv", index=False)
# edit_privacy_df.to_csv("data/Final_data/accessibility_correction_synthetic_trajectories.csv", index=False)


transition_distance=pd.read_csv("data/firstpresentation/resultcsvV1.csv")
print(transition_distance['user_id'].unique())
print(transition_distance)
edit_synthetic_df=transition_distance.loc[transition_distance['user_id'].isin([53, 75, 76, 86, 100, 104, 107, 110, 175])]


# Remove rows with specified user IDs
edit_synthetic_df = transition_distance[~transition_distance['user_id'].isin([53, 75, 76, 86, 100, 104, 107, 110, 175])]
print(edit_synthetic_df)
edit_synthetic_df.to_csv("data/firstpresentation/resultcsvV1.csv", index=False)