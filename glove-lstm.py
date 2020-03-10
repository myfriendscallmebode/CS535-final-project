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
import bcolz

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

def create_emb_layer(weights_matrix, non_trainable=False):

        num_embeddings,embedding_dim = weights_matrix.shape
        

        emb_layer = nn.Embedding(num_embeddings,embedding_dim)
        #loading the weights back to the embbedding 
        
        emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer,num_embeddings,embedding_dim

class Net(nn.Module):

    
    def __init__(self,weights_matrix, hidden_size,embedding_size, padding_idx):
        super(Net, self).__init__()
        # Using helper function to define the weight
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        #self.embedding = nn.Embedding(input_size, embedding_size, padding_idx,pretrained=pretrained_embedding) #learns embedding, input size is n_words, maybe experiment with different representation
        self.lstm = nn.LSTM(input_size = embedding_dim, 
        					hidden_size = hidden_size,
        					num_layers=2) #trying different nuymber of layers
        self.fc1 = nn.Linear(hidden_size + 1, 20) #try different number of hidden units?
        self.fc2 = nn.Linear(20, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.n_layers = 2



    #forward pass
    def forward(self, batch, tag, hidden):
    	#print("batch size",batch.size())
    	embed = self.embedding(batch).view(1, BATCH_SIZE, -1) #embedding of vector
    	output = embed
    	#print("output size",output.size())
    	#print("hidden size",hidden.size())
    	output, hidden = self.lstm(output, hidden) #GRU layer
    	output = torch.cat((output.squeeze(), tag), dim = 1) #concatinate hashtag count
    	output = self.leakyrelu(self.fc1(output)) #one fully connected layer
    	output = self.fc2(output) #second FC layer
    	return output, hidden

    #initializes the hidden state for a new tensor input
    #maybe we can learn initial states?
    def init_hidden(self, batch_size):
        #return torch.zeros(1, batch_size, self.hidden_size, device=device) #change first dimension for number of hidden layers of gru
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden
#evaluates the network, similar to assignment 3
#
def eval_batch(batch_tweets, batch_tag, batch_label):
    net.eval() 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    hidden = net.init_hidden(BATCH_SIZE)
    net.zero_grad()
    for i in range(batch_tweets.size()[1]): #forward pass
    	output, hidden = net(batch_tweets[:,i], batch_tag, hidden)
    values, predicted = torch.max(output.data, 1) #index of highest energy, needs to squeeze it for dimension sake
    #total = batch_label.size(0)
    correct = (predicted == batch_label.squeeze().data).sum()
    #print(correct.float()/total)
    loss = criterion(output, batch_label.squeeze())
    net.train()
    return loss, correct.float()

#trains network
def train(batch_tweets, batch_tag, batch_label):
    hidden = net.init_hidden(BATCH_SIZE) #initialize hidden state per tweet
    net.zero_grad()
    for i in range(batch_tweets.size()[1]): #forward pass
    	output, hidden = net(batch_tweets[:,i], batch_tag, hidden) #word by word input, don't know any other way to train it
    #print(output.size())
    #print(batch_label.size())
    loss = criterion(output, batch_label.squeeze())
    loss.backward() #backward pass
    optimizer.step()
    return output, loss.item()



if __name__ == '__main__':
	device = "cuda" #make sure on GPU
	BATCH_SIZE = 250
	NUM_EPOCHS = 50

	#Open data dictionary, made with clean.py
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)
	vectors = bcolz.open('6B.50.dat')[:]
	words = pickle.load(open('6B.50_words.pkl', 'rb'))
	word2idx = pickle.load(open('6B.50_idx.pkl', 'rb'))


    # making glove dictionary 

	glove = {w: vectors[word2idx[w]] for w in words}

	#separate
	tweets_train = data_dict['tweets_train']
	hashtags_train = data_dict['hashtags_train']
	labels_train = data_dict['labels_train']
	tweets_test = data_dict['tweets_test']
	hashtags_test = data_dict['hashtags_test']
	labels_test = data_dict['labels_test']
	tweets_brand = data_dict['tweets_brand']

	#initialize language, adds each tweet to the language
	twitter_lang = Lang()
	for tweet in tweets_train: #make language from all tweets
		twitter_lang.add_tweet(tweet)
	for tweet in tweets_test:
		twitter_lang.add_tweet(tweet)
	for tweet in tweets_brand:
		twitter_lang.add_tweet(tweet)

	pad_index = twitter_lang.word2index['<pad>']

	# initializing weights matrix

	matrix_len = len(twitter_lang.word2index)
	weights_matrix = np.zeros((matrix_len, 50))
	words_found = 0

	
	# If we find the embedding in glove then use that or else initial random vector for that word

	for i, word in enumerate(twitter_lang.word2index):
		try: 
			weights_matrix[i] = glove[word]
			words_found += 1
		except KeyError:
			weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))
	#print("wf",words_found)

	#print('weights_matrix',weights_matrix)

	#initialize network
	net = Net(weights_matrix = weights_matrix, 
				hidden_size = 100, 
				embedding_size = 50,
				padding_idx =  pad_index).cuda() #hidden size is arbitrary for now
	criterion = nn.CrossEntropyLoss() #maybe experiment with different loss. NLLL?
	optimizer = optim.Adam(net.parameters()) #weight decay??

	#lists of tensors basically
	tweets_train = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_train], 0)
	tweets_test = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_test], 0)

	labels_train = torch.stack([torch.tensor([label]).cuda() for label in labels_train], 0)
	labels_test = torch.stack([torch.tensor([label]).cuda() for label in labels_test], 0)

	hashtags_train = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_train], 0) #just using number of hashtags used in tweet
	hashtags_test = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_test], 0)

	#if you want to pretrain
	#pretrained_dict = torch.load("mytraining.pth") 
	net.train()
	#model_dict = net.state_dict(pretrained_dict) 

	acc_matrix = np.zeros((NUM_EPOCHS, 2))
	loss_matrix = np.zeros((NUM_EPOCHS, 2))

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
			#print('output',output.shape)

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
		acc_matrix[epoch, 0] = train_acc
		loss_matrix[epoch, 0] = train_loss


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
		acc_matrix[epoch, 1] = test_acc
		loss_matrix[epoch, 1] = test_loss

		print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
	torch.save(net.state_dict(), 'glove-lstm.pth') #save state dictionary
	np.savetxt("glovelstm_acc.csv", acc_matrix, delimiter=",")
	np.savetxt("glovelstm_loss.csv", loss_matrix, delimiter=",")




