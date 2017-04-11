import pandas as pd
data = pd.read_csv('./data/news_data.csv')
vox = data.loc[data.org =='vox'].sample(1000)
jez = data.loc[data.org != 'vox'].sample(1000)
sampled = pd.concat([vox, jez], ignore_index=True)
sampled.columns = data.columns
sampled.to_json('./data/news_data_sample.json', orient='records')