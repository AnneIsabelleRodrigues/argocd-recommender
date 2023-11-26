import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

file_path1 = os.environ.get('dev-dataset1-path')
file_path2 = os.environ.get('dev-dataset2-path')
file_paths = os.environ.get('dev-datasetsongs-path')


playlist = pd.read_csv(file_path1, header=0)
playlist2 = pd.read_csv(file_path2, header=0)
songs = pd.read_csv(file_paths, header=0)

playlist = pd.concat([playlist, playlist2], ignore_index=True)

songs.reset_index(inplace=True)
result = pd.merge(playlist, songs, on='track_name')

result['track_name'] = result['track_name'].apply(lambda x: [str(x)])
data = result.groupby(['pid'])['track_name'].agg("sum")
data = pd.DataFrame(data)
data.dropna(inplace=True)
data.reset_index(inplace=True)

transactions = data['track_name'].tolist()
data['pid'] = data['pid'].astype(int)

transaction_encoder = TransactionEncoder()
transaction_array = transaction_encoder.fit(transactions).transform(transactions)
transaction_dataframe = pd.DataFrame(transaction_array, columns=transaction_encoder.columns_)
itemsets=fpgrowth(transaction_dataframe,min_support=0.02, use_colnames=True)
itemsets.sort_values("support", ascending=False)

association_rules_df = association_rules(itemsets, metric="confidence", min_threshold=.50)

association_rules_df = association_rules_df[(association_rules_df["support"]>0.05) & (association_rules_df["confidence"]>0.1) & (association_rules_df["lift"]>5)].sort_values("confidence", ascending=False)

association_rules_df.to_pickle("/modelo/rules.pkl")
data.to_pickle("/modelo/data.pkl")