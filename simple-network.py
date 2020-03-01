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

#to convert language to one-hot vectors
#taken from pytorch tutorial
class Lang:
    def __init__(self):
    	# I don't think we will need SOS and EOS
        self.word2index = {}
        self.word2count = {}
        self.index2word = {} 
        self.n_words = 0

    def addTweet(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromText(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size = hidden_size, 
        					hidden_size = hidden_size)
        self.fc = nn.Linear(hidden_size, 3)


    def forward(self, tweet, hidden):
    	#print(hidden.size())
    	embedded = self.embedding(tweet).view(1, 1, -1)
    	output = embedded
    	#print(output.size())
    	output, hidden = self.gru(output, hidden)
    	output = self.fc(output)
    	return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def eval_net(data, targets):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for tweet, label in zip(data, targets):
    	hidden = net.initHidden()
    	net.zero_grad()
    	for i in range(tweet.size()[0]):
        	output, hidden = net(tweet[i], hidden)
    	values, predicted = torch.max(output.squeeze(), 0) #index of highest energy
    	total += 1
    	correct += (predicted == label)
    	loss = criterion(output.view(1,3), label)
    	total_loss += loss.item()
    net.train()
    return total_loss / total, correct.float() / total

def train(tweet, label):
    hidden = net.initHidden()
    net.zero_grad()
    for i in range(tweet.size()[0]):
        output, hidden = net(tweet[i], hidden)
    loss = criterion(output.view(1,3), label)
    loss.backward()
    optimizer.step()
    return output, loss.item()

if __name__ == '__main__':
	device = "cuda"
	#data
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	text = data_dict['text']
	hashags = data_dict['hashtags']
	labels = data_dict['gender']

	#initialize language
	twitter_lang = Lang()
	for tweet in text:
		twitter_lang.addTweet(tweet)

	#training
	net = Net(input_size = twitter_lang.n_words, hidden_size = 100).cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters())

	#If you don't want to train it all
	index_train = np.random.choice(len(text), size =  15000)
	index_test = [i for i in range(len(text)) if i not in index_train]
	text_train = [text[i] for i in index_train]
	labels_train = [labels[i] for i in index_train]
	text_test = [text[i] for i in index_test]
	labels_test = [labels[i] for i in index_test]

	text_train = [tensorFromText(twitter_lang, tweet).cuda() for tweet in text_train]
	text_test = [tensorFromText(twitter_lang, tweet).cuda() for tweet in text_test]

	labels_train = [torch.tensor([label]).cuda() for label in labels_train]
	labels_test = [torch.tensor([label]).cuda() for label in labels_test]

	
	#pretrained_dict = torch.load("mytraining.pth") 
	net.train()
	#model_dict = net.state_dict(pretrained_dict) 

	print("training on tweets...")
	for epoch in range(5): 
		iters = 1
		for tweet, target in zip(text_train, labels_train):
			if iters%1000 == 0:
				print("iteration:  " + str(iters))
			net.zero_grad()
			output, loss = train(tweet, target)
			iters += 1

		print('    Finish training this EPOCH, start evaluating...')
		train_loss, train_acc = eval_net(text_train, labels_train)
		test_loss, test_acc = eval_net(text_test, labels_test)
		print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
	torch.save(net.state_dict(), 'mytraining.pth')
	    #



