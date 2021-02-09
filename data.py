import pandas as pd
import numpy as np

header = ['user_id','item_id','rating','timestamp']
data = pd.read_csv('ratings_Toys_and_Games.csv',names=header)
#data = data.groupby('user_id').filter(lambda x: len(x) > 10)
#data.to_csv('amazon_toys_filter.csv',index=False)