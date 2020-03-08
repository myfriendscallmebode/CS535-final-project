import pandas as pd  
import numpy as np
import emoji
import re
import pickle

df = pd.read_csv("tweet-data.csv", encoding='latin1')
df = df[df['gender'].isin(['male','female','brand'])]
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

def pad(tweets): #padding
	split_tweets = [tweet.split() for tweet in tweets]
	max_len = max(map(len, split_tweets))
	for tweet in split_tweets:
		for i in range(len(tweet), max_len):
			tweet.append("<pad>")
	result = [' '.join(tweet) for tweet in split_tweets]
	return(result)

tweets, hashtags = map(list, zip(*[clean(t) for t in tweets]))

tweets = pad(tweets)

#split into train/test
index_train = np.random.choice(len(tweets), size =  15000, replace = False) #arbitrary split
index_test = [i for i in range(len(tweets)) if i not in index_train]
index_brand = [i for i in range(len(tweets)) if gender[i] == 2]
# remove brands from train set
index = []
for i in index_train:
	if gender[i] != 2:
		index.append(i)
tweets_train = [tweets[i] for i in index]
labels_train = [gender[i] for i in index]
hashtags_train = [hashtags[i] for i in index]
print("train size")
print(len(index))
# male-female test set
index = []
for i in index_test:
	if gender[i] != 2:
		index.append(i)
tweets_test = [tweets[i] for i in index]
labels_test = [gender[i] for i in index]
hashtags_test = [hashtags[i] for i in index]
print("test size")
print(len(index))
# brand test set
tweets_brand = [tweets[i] for i in index_brand]
labels_brand = [gender[i] for i in index_brand]
hashtags_brand = [hashtags[i] for i in index_brand]
print("brand size")
print(len(index_brand))





data_dict = {'tweets_train': tweets_train,
			   'hashtags_train': hashtags_train,
			   'labels_train': labels_train,
			   'tweets_test': tweets_test,
			   'hashtags_test': hashtags_test,
			   'labels_test': labels_test,
			   'tweets_brand': tweets_brand,
			   'hashtags_brand': hashtags_brand,
			   'labels_brand': labels_brand}


with open('data-dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#to open
# with open('data-dict.pickle', 'rb') as handle:
# 	new_dict = pickle.load(handle)
