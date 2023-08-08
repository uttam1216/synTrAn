import pandas as pd
df=pd.read_csv('mobility_map.csv',index_col=0)
print(df)
final_df=df.sort_values(by=['Segments'])
final_df=final_df.reset_index()
print(final_df)

final_df.to_csv("Final_training_csv/mobility_map_train.csv",index=False)