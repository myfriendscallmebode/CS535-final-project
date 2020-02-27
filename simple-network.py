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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self, x):
        return x

if __name__ == '__main__':
	tokenizer = get_tokenizer("basic_english")
	#data
	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	text = data_dict['text']
	hashags = data_dict['hashtags']
	labels = data_dict['gender']

	print(tokenizer(text[0]))


	net = Net().cuda()





