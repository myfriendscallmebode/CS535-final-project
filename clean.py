import pandas as pd  
import numpy as np
import emoji

df = pd.read_csv("tweet-data.csv")
print(df.head())




#tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())