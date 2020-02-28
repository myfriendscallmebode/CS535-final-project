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
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for tweet, label in zip(data, targets):
    	hidden = net.initHidden()
    	tweet_in = tensorFromText(twitter_lang, tweet).cuda()
    	target = torch.tensor([label]).cuda()
    	hidden = net.initHidden()
    	net.zero_grad()
    	for i in range(tweet_in.size()[0]):
        	output, hidden = net(tweet_in[i], hidden)
    	values, predicted = torch.max(output.squeeze(), 0) #index of highest energy
    	total += 1
    	#print(predicted) #predicted not working
    	#print(target.squeeze())
    	correct += (predicted == target)
    	loss = criterion(output.view(1,3), target)
    	total_loss += loss.item()
    net.train() # Why would I do this?
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
	optimizer = optim.SGD(net.parameters(), lr=0.001)

	index_subset = np.random.choice(len(text), size = 3000)
	text_subset = [text[i] for i in index_subset]
	label_subset = [labels[i] for i in index_subset]
	print("training on tweets...")
	for epoch in range(10): 
		iters = 1
		for tweet, label in zip(text_subset, label_subset):
			if iters%500 == 0:
				print("iteration:  " + str(iters))
			net.zero_grad()
			tweet_in = tensorFromText(twitter_lang, tweet).cuda()
			target = torch.tensor([label]).cuda()
			output, loss = train(tweet_in, target)
			iters += 1

		print('    Finish training this EPOCH, start evaluating...')
		train_loss, train_acc = eval_net(text_subset, label_subset)
		print('EPOCH: %d train_loss: %.5f train_acc: %.5f' %
              (epoch+1, train_loss, train_acc))
	    #print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
        #      (epoch+1, train_loss, train_acc, test_loss, test_acc))




