import pandas as pd  
import numpy as np
import emoji
import re
import pickle

df = pd.read_csv("tweet-data.csv", encoding='latin1')
tweets = df['text'].tolist()
gender = df['gender'].tolist()
gender = [0 if g == 'male' else 1 if g == 'female' else 2 for g in gender]

def clean(tweet):
	tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split()) #url
	tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split()) #punctuation
	tweet = tweet.lower() #lowercase
	tweet = emoji.demojize(tweet) #turn emojis into words
	tweet = ' '.join(tweet.split())
	tweet = re.sub('[^A-Za-z0-9 #@:]+', '', tweet) #punctuation/special characters again, keeps emojis like :smiley:, keeps hashtags and @'s
	tweet = ' '.join( [w for w in tweet.split() if len(w)>1 or w in ["a", "i"]] ) #remove single letters except a and i
	hashtags = re.findall(r"#(\w+)", tweet) #extract hashtags
	mentions = re.findall(r"@(\w+)", tweet) #extract mentions
	tweet = ' '.join(re.sub(r"#(\w+)", "", tweet).split()) #remove hashtags
	tweet = ' '.join(re.sub(r"@(\w+)", "<mention>", tweet).split()) #replace a specific mention with '<mention>' token
	return(tweet, hashtags)

tweets, hashtags = map(list, zip(*[clean(t) for t in tweets]))

data_dict = {'text': tweets,
			   'hashtags': hashtags,
			   'gender': gender}


with open('data-dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#to open
# with open('data-dict.pickle', 'rb') as handle:
# 	new_dict = pickle.load(handle)
