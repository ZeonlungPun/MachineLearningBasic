#page rank2
import pandas as pd
data=pd.read_excel('influence.xlsx',engine='openpyxl')
freq_matrix=data.pivot(columns='influencer_id',index='follower_id',values='num')#投票矩阵
freq_matrix.fillna(0,inplace=True)
print(freq_matrix)