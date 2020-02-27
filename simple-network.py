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
    def __init__(self, name):
    	# I don't think we will need SOS and EOS
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {} 
        self.n_words = 0

    def addSentence(self, sentence):
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

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



if __name__ == '__main__':
	#data
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	text = data_dict['text']
	hashags = data_dict['hashtags']
	labels = data_dict['gender']

	language = Lang()
	for tweet in text:
		language.addSentence(tweet)




	net = Net().cuda()





