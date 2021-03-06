from __future__ import print_function
from __future__ import division
import torch
import torchtext
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14
import numpy as np
import pickle
from torchtext.data.utils import get_tokenizer

#taken from pytorch tutorial
#class that keeps dictionaries of words to their respective indices for one-hot vectors
#It also counts the words and keeps a dictionary to go backwards as well
class Lang:
    def __init__(self):
    	# I don't think we will need SOS and EOS
        self.word2index = {}
        self.word2count = {}
        self.index2word = {} 
        self.n_words = 0

    #adds a single tweet to the language
    def add_tweet(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    #adds a single word to the language, helper for add_tweet
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#Given a language and sentence, converts to a numeric vector
#helper for tensor_from_text
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#given a language and sentence, converts to a tensor
def tensor_from_text(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class Net(nn.Module):
    def __init__(self, vocab_size, seq_length, embedding_size, hidden_size, padding_idx):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx) #learns embedding, input size is n_words, maybe experiment with different representation
        self.gru = nn.GRU(input_size = seq_length*embedding_size, 
        					hidden_size = hidden_size,
        					num_layers=1) #trying different nuymber of layers
        self.fc1 = nn.Linear(hidden_size + 1, 20) #try different number of hidden units?
        self.fc2 = nn.Linear(20, 2) #change to 2 labels
        self.leakyrelu = nn.LeakyReLU()

    #forward pass
    def forward(self, batch, tag, hidden):
    	embed = self.embedding(batch).view(1, BATCH_SIZE, -1) #embedding of vector
    	output = embed
    	output, hidden = self.gru(output, hidden) #GRU layer
    	output = torch.cat((output.squeeze(), tag), dim = 1) #concatinate hashtag count
    	output = self.leakyrelu(self.fc1(output)) #one fully connected layer
    	output = self.fc2(output) #second FC layer
    	return output, hidden

    #initializes the hidden state for a new tensor input
    #maybe we can learn initial states?
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device) #change first dimension for number of hidden layers of gru

#evaluates the network, similar to assignment 3
#
def eval_batch(batch_tweets, batch_tags, batch_labels):
	net.eval() 
	hidden = net.init_hidden(BATCH_SIZE) #initialize hidden state per tweet
	criterion = nn.CrossEntropyLoss(reduction='mean')
	net.zero_grad()
	output, hidden = net(batch_tweets, batch_tags, hidden)
	values, predicted = torch.max(output.data, 1) #index of highest energy, needs to squeeze it for dimension sake
	correct = (predicted == batch_labels.squeeze().data).sum()
	loss = criterion(output, batch_labels.squeeze())
	net.train()
	return loss, correct.float()

#trains network
def train(batch_tweets, batch_tags, batch_labels):
	hidden = net.init_hidden(BATCH_SIZE) #initialize hidden state per tweet
	net.zero_grad()
	output, hidden = net(batch_tweets, batch_tags, hidden)
	loss = criterion(output, batch_labels.squeeze())
	loss.backward() #backward pass
	optimizer.step()
	return output, loss.item()

if __name__ == '__main__':
	device = "cuda" #make sure on GPU
	BATCH_SIZE = 100
	NUM_EPOCHS = 100

	writer = SummaryWriter(log_dir='./runs')

	#Open data dictionary, made with clean.py
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	#separate
	tweets_train = data_dict['tweets_train']
	hashtags_train = data_dict['hashtags_train']
	labels_train = data_dict['labels_train']
	tweets_test = data_dict['tweets_test']
	hashtags_test = data_dict['hashtags_test']
	labels_test = data_dict['labels_test']

	#initialize language, adds each tweet to the language
	twitter_lang = Lang()
	for tweet in tweets_train: #make language from all tweets
		twitter_lang.add_tweet(tweet)
	for tweet in tweets_test:
		twitter_lang.add_tweet(tweet)

	pad_index = twitter_lang.word2index['<pad>']

	tweets_train = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_train], 0)
	tweets_test = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_test], 0)

	labels_train = torch.stack([torch.tensor([label]).cuda() for label in labels_train], 0)
	labels_test = torch.stack([torch.tensor([label]).cuda() for label in labels_test], 0)

	hashtags_train = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_train], 0) #just using number of hashtags used in tweet
	hashtags_test = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_test], 0)

	seq_length = tweets_train.size()[1]

	#initialize network
	net = Net(vocab_size = twitter_lang.n_words, 
				seq_length = seq_length,
				embedding_size = 10,
				hidden_size = 50,
				padding_idx =  pad_index).cuda() #hidden size is arbitrary for now
	criterion = nn.CrossEntropyLoss() #maybe experiment with different loss. NLLL?
	optimizer = optim.Adam(net.parameters()) #weight decay??


	#if you want to pretrain
	#pretrained_dict = torch.load("mytraining.pth") 
	net.train()
	#model_dict = net.state_dict(pretrained_dict) 

	#now we're training
	print("training on tweets...")
	for epoch in range(NUM_EPOCHS): 
		permutation = torch.randperm(tweets_train.size()[0]) #shuffle batches
		iters = 1

		for i in range(0,tweets_train.size()[0] - tweets_train.size()[0]%BATCH_SIZE, BATCH_SIZE):
			optimizer.zero_grad()
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			output, loss = train(batch_tweets, batch_hashtags, batch_labels)

		#eval training data
		correct = 0
		total = 0
		total_loss = 0
		for i in range(0,tweets_train.size()[0] - tweets_train.size()[0]%BATCH_SIZE, BATCH_SIZE):
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			loss, num_correct = eval_batch(batch_tweets, batch_hashtags, batch_labels)
			total += batch_labels.size(0)
			total_loss += loss.item()
			correct += num_correct
		train_loss = total_loss / total
		train_acc = correct / total

		#eval test data
		correct = 0
		total = 0
		total_loss = 0
		#print(tweets_test.size()[0]%BATCH_SIZE)
		for i in range(0,tweets_test.size()[0] - tweets_test.size()[0]%BATCH_SIZE, BATCH_SIZE):
			batch_tweets, batch_hashtags, batch_labels = tweets_test[i:i+BATCH_SIZE], hashtags_test[i:i+BATCH_SIZE], labels_test[i:i+BATCH_SIZE]
			loss, num_correct = eval_batch(batch_tweets, batch_hashtags, batch_labels)
			total += batch_labels.size(0)
			total_loss += loss.item()
			correct += num_correct
		test_loss = total_loss / total
		test_acc = correct / total

		print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
	#torch.save(net.state_dict(), 'mytraining.pth') #save state dictionary
	net.eval()
	batch_tweets, batch_hashtags, batch_labels = tweets_train[0:BATCH_SIZE], hashtags_train[0:BATCH_SIZE], labels_train[0:BATCH_SIZE]
	hidden = net.init_hidden(BATCH_SIZE)
	#writer.add_graph(net, input_to_model=(batch_tweets[:,0], batch_hashtags, hidden))
	#writer.close()
