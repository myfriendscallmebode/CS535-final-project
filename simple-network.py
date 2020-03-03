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
        for word in sentence.split(' '):
            self.add_word(word)

    #adds a single word to the language, helper for addTweet
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
#I regret using camelCase
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#given a language and sentence, converts to a tensor
def tensor_from_text(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size) #learns embedding, input size is n_words, maybe experiment with different representation
        self.gru = nn.GRU(input_size = hidden_size, 
        					hidden_size = hidden_size)
        self.fc1 = nn.Linear(hidden_size, 50) #for more parameters
        self.fc2 = nn.Linear(50, 3)

    #forward pass
    def forward(self, word, hidden):
    	embedded = self.embedding(word).view(1, 1, -1) #embedding of vector
    	output = embedded
    	output, hidden = self.gru(output, hidden) #GRU layer
    	output = self.fc1(output) #one fully connected layer
    	output = self.fc2(output) #second FC layer
    	return output, hidden

    #initializes the hidden state for a new tensor input
    #maybe we can learn initial states?
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#evaluates the network, similar to assignment 3
#takes a while since it goes through every tweet
def eval_net(data, targets):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for tweet, label in zip(data, targets):
    	hidden = net.init_hidden()
    	net.zero_grad()
    	for i in range(tweet.size()[0]):
        	output, hidden = net(tweet[i], hidden)
    	values, predicted = torch.max(output.squeeze(), 0) #index of highest energy, needs to squeeze it for dimension sake
    	total += 1
    	correct += (predicted == label)
    	loss = criterion(output.view(1,3), label)
    	total_loss += loss.item()
    net.train()
    return total_loss / total, correct.float() / total

#trains network
def train(tweet, label):
    hidden = net.init_hidden() #initialize hidden state per tweet
    net.zero_grad()
    for i in range(tweet.size()[0]): #forward pass
        output, hidden = net(tweet[i], hidden) #word by word input, don't know any other way to train it
    loss = criterion(output.view(1,3), label)
    loss.backward() #backward pass
    optimizer.step()
    return output, loss.item()

if __name__ == '__main__':
	device = "cuda" #make sure on GPU
	#Open data dictionary, made with clean.py
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	#separate
	text = data_dict['text']
	hashtags = data_dict['hashtags'] #still yet to use these, I want to though, add them at a linear layer or something
	labels = data_dict['gender']

	#initialize language, adds each tweet to the language
	twitter_lang = Lang()
	for tweet in text:
		twitter_lang.add_tweet(tweet)

	#initialize network
	net = Net(input_size = twitter_lang.n_words, hidden_size = 100).cuda() #hidden size is arbitrary for now
	criterion = nn.CrossEntropyLoss() #maybe experiment with differne loss. NLLL?
	optimizer = optim.Adam(net.parameters(), weight_decay = 0.01) #weight decay??

	#Arbitrary split of train/validation
	index_train = np.random.choice(len(text), size =  15000)
	index_test = [i for i in range(len(text)) if i not in index_train]
	text_train = [text[i] for i in index_train]
	labels_train = [labels[i] for i in index_train]
	text_test = [text[i] for i in index_test]
	labels_test = [labels[i] for i in index_test]

	#lists of tensors basically
	text_train = [tensor_from_text(twitter_lang, tweet).cuda() for tweet in text_train]
	text_test = [tensor_from_text(twitter_lang, tweet).cuda() for tweet in text_test]

	labels_train = [torch.tensor([label]).cuda() for label in labels_train]
	labels_test = [torch.tensor([label]).cuda() for label in labels_test]

	#if you want to pretrain
	pretrained_dict = torch.load("mytraining.pth") 
	net.train()
	model_dict = net.state_dict(pretrained_dict) 

	#now we're training
	print("training on tweets...")
	for epoch in range(20): 
		iters = 1
		for tweet, target in zip(text_train, labels_train):
			if iters%1000 == 0: #prints which iteration it's on for timekeeping sake
				print("iteration:  " + str(iters))
			net.zero_grad()
			output, loss = train(tweet, target)
			iters += 1

		print('    Finish training this EPOCH, start evaluating...')
		train_loss, train_acc = eval_net(text_train, labels_train)
		test_loss, test_acc = eval_net(text_test, labels_test)
		print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
	torch.save(net.state_dict(), 'mytraining.pth') #save state dictionary
	    #



