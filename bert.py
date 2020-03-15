from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14
import numpy as np
import matplotlib.pyplot as plt
from transformers import *
import pickle

class Net(BertPreTrainedModel):
	def __init__(self, pretrained_weights):
		super(Net, self).__init__(BertConfig.from_pretrained(pretrained_weights))
		self.bert = BertModel.from_pretrained(pretrained_weights) #pretrained bert
		self.dropout = torch.nn.Dropout(p=0.1, inplace=False) #taken from bert
		self.fc1 =  nn.Linear(in_features = 768 + 1, out_features = 50, bias=True) #number of bert output layers
		self.fc2 = nn.Linear(in_features = 50, out_features = 3, bias=True)

    #forward pass
	def forward(self, input_ids, input_tags):
		_, output = self.bert(input_ids)
		output = self.dropout(output) #what the original 2 class classifier did
		output = torch.cat((output.squeeze(), input_tags), dim = 1) #concatinate hashtag count
		output = self.fc1(output)
		output = self.fc2(output)
		return output


def get_input_ids(tweets, tokenizer):
	input_ids = torch.tensor(tokenizer.batch_encode_plus(tweets, add_special_tokens=True,  pad_to_max_length=True)['input_ids']).cuda()
	return input_ids


def eval_batch(batch_tweets, batch_tags, batch_labels):
	net.eval() 
	criterion = nn.CrossEntropyLoss(reduction='mean')
	net.zero_grad()
	output = net(batch_tweets, batch_tags)
	values, predicted = torch.max(output.data, 1) #index of highest energy, needs to squeeze it for dimension sake
	correct = (predicted == batch_labels.squeeze().data).sum()
	loss = criterion(output, batch_labels.squeeze())
	net.train()
	return loss, correct.float()

#trains network
def train(batch_tweets, batch_tags, batch_labels):
	net.zero_grad()
	output = net(batch_tweets, batch_tags)
	loss = criterion(output, batch_labels.squeeze())
	loss.backward() #backward pass
	optimizer.step()
	return output, loss.item()


if __name__ == '__main__':
	#print(torch.cuda.memory_summary())
	device = "cuda" #make sure on GPU
	BATCH_SIZE = 50
	NUM_EPOCHS = 10

	pretrained_weights = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
	net = Net(pretrained_weights).cuda()
	#print(net)
	criterion = nn.CrossEntropyLoss() #maybe experiment with different loss. NLLL?
	optimizer = optim.Adam(net.parameters()) #weight decay??

	with open('data-dict.pickle', 'rb') as handle:
		data_dict = pickle.load(handle)

	#separate
	tweets_train = data_dict['tweets_train']
	hashtags_train = data_dict['hashtags_train']
	labels_train = data_dict['labels_train']
	tweets_test = data_dict['tweets_test']
	hashtags_test = data_dict['hashtags_test']
	labels_test = data_dict['labels_test']


	tweets_train = get_input_ids(tweets_train, tokenizer)
	tweets_test = get_input_ids(tweets_test, tokenizer)

	labels_train = torch.stack([torch.tensor([label]).cuda() for label in labels_train], 0)
	labels_test = torch.stack([torch.tensor([label]).cuda() for label in labels_test], 0)

	hashtags_train = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_train], 0) #just using number of hashtags used in tweet
	hashtags_test = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_test], 0)

	#if you want to pretrain
	#pretrained_dict = torch.load("mytraining.pth") 
	net.train()
	#model_dict = net.state_dict(pretrained_dict) 

	#now we're training
	#print(torch.cuda.memory_summary())
	print("training on tweets...")
	for epoch in range(NUM_EPOCHS): 
		permutation = torch.randperm(tweets_train.size()[0]) #shuffle batches
		iters = 1

		for i in range(0,tweets_train.size()[0], BATCH_SIZE):
			optimizer.zero_grad()
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			output, loss = train(batch_tweets, batch_hashtags, batch_labels)

		#eval training data
		correct = 0
		total = 0
		total_loss = 0
		for i in range(0,tweets_test.size()[0], BATCH_SIZE):
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
	torch.save(net.state_dict(), 'bert-pretrained.pth') #save state dictionary
	


	