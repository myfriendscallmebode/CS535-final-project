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
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

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
    def __init__(self, input_size, embedding_size, hidden_size, padding_idx):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx) #learns embedding, input size is n_words, maybe experiment with different representation
        self.lstm = nn.LSTM(input_size = embedding_size, 
        					hidden_size = hidden_size,
        					num_layers=2) #trying different nuymber of layers
        self.fc1 = nn.Linear(hidden_size + 1, 20) #try different number of hidden units?
        self.fc2 = nn.Linear(20, 2)
        self.leakyrelu = nn.LeakyReLU()
        self.n_layers = 2

    #forward pass
    def forward(self, batch, tag, hidden):
    	for i in range(batch.size()[1]): #forward pass
    		embed = self.embedding(batch[:,i]).view(1, BATCH_SIZE, -1) #embedding of vector
    		output = embed
    		self.lstm.flatten_parameters()
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

# define priors
def model(batch, tag, hidden, label):
	log_softmax = nn.LogSoftmax(dim=1)
	    
	embedding_prior = Normal(loc=pretrained_dict['embedding.weight'], scale=torch.ones_like(net.embedding.weight))

	lstmih0w_prior = Normal(loc=pretrained_dict['lstm.weight_ih_l0'], scale=torch.ones_like(net.lstm.weight_ih_l0))
	lstmih0b_prior = Normal(loc=pretrained_dict['lstm.bias_ih_l0'], scale=torch.ones_like(net.lstm.bias_ih_l0))
	lstmhh0w_prior = Normal(loc=pretrained_dict['lstm.weight_hh_l0'], scale=torch.ones_like(net.lstm.weight_hh_l0))
	lstmhh0b_prior = Normal(loc=pretrained_dict['lstm.bias_hh_l0'], scale=torch.ones_like(net.lstm.bias_hh_l0))

	lstmih1w_prior = Normal(loc=pretrained_dict['lstm.weight_ih_l1'], scale=torch.ones_like(net.lstm.weight_ih_l1))
	lstmih1b_prior = Normal(loc=pretrained_dict['lstm.bias_ih_l1'], scale=torch.ones_like(net.lstm.bias_ih_l1))
	lstmhh1w_prior = Normal(loc=pretrained_dict['lstm.weight_hh_l1'], scale=torch.ones_like(net.lstm.weight_hh_l1))
	lstmhh1b_prior = Normal(loc=pretrained_dict['lstm.bias_hh_l1'], scale=torch.ones_like(net.lstm.bias_hh_l1))

	fc1w_prior = Normal(loc=pretrained_dict['fc1.weight'], scale=torch.ones_like(net.fc1.weight))
	fc1b_prior = Normal(loc=pretrained_dict['fc1.bias'], scale=torch.ones_like(net.fc1.bias))
	    
	fc2w_prior = Normal(loc=pretrained_dict['fc2.weight'], scale=torch.ones_like(net.fc2.weight))
	fc2b_prior = Normal(loc=pretrained_dict['fc2.bias'], scale=torch.ones_like(net.fc2.bias))
	    
	priors = {'embedding.weight': embedding_prior, 'lstm.weight_ih_l0': lstmih0w_prior, 'lstm.bias_ih_l0': lstmih0b_prior,
	'lstm.weight_hh_l0': lstmhh0w_prior, 'lstm.bias_hh_l0': lstmhh0b_prior, 'lstm.weight_ih_l0': lstmih0w_prior,
	'lstm.bias_ih_l0': lstmih0b_prior,'lstm.weight_hh_l0': lstmhh0w_prior, 'lstm.bias_hh_l0': lstmhh0b_prior,
	'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
	    
	# lift module parameters to random variables sampled from the priors
	lifted_module = pyro.random_module("module", net, priors)
	# sample module
	lifted_model = lifted_module()

	output, hidden = lifted_model(batch, tag, hidden)
	    
	lhat = log_softmax(output)
	    
	pyro.sample("obs", Categorical(logits=lhat), obs=label)

# variational distribution (approximate posterior)
def guide(batch, tag, hidden, label):
	softplus = torch.nn.Softplus()
    
	# embedding weight distribution priors
	embedding_mu = torch.randn_like(net.embedding.weight)
	embedding_sigma = torch.randn_like(net.embedding.weight)
	embedding_mu_param = pyro.param("embedding_mu", embedding_mu)
	embedding_sigma_param = softplus(pyro.param("embedding_sigma", embedding_sigma))
	embedding_prior = Normal(loc=embedding_mu_param, scale=embedding_sigma_param)
	# lstm layer 1 input-hidden weight distribution priors
	lstmih0w_mu = torch.randn_like(net.lstm.weight_ih_l0)
	lstmih0w_sigma = torch.randn_like(net.lstm.weight_ih_l0)
	lstmih0w_mu_param = pyro.param("lstmih0w_mu", lstmih0w_mu)
	lstmih0w_mu_param  = softplus(pyro.param("lstmih0w_sigma", lstmih0w_sigma))
	lstmih0w_prior = Normal(loc=lstmih0w_mu_param , scale=lstmih0w_mu_param )
	# lstm layer 1 input-hidden bias distribution priors
	lstmih0b_mu = torch.randn_like(net.lstm.bias_ih_l0)
	lstmih0b_sigma = torch.randn_like(net.lstm.bias_ih_l0)
	lstmih0b_mu_param = pyro.param("lstmih0b_mu", lstmih0b_mu)
	lstmih0b_sigma_param = softplus(pyro.param("lstmih0b_sigma", lstmih0b_sigma))
	lstmih0b_prior = Normal(loc=lstmih0b_mu_param, scale=lstmih0b_sigma_param)
	# lstm layer 1 hidden-hidden weight distribution priors
	lstmhh0w_mu = torch.randn_like(net.lstm.weight_hh_l0)
	lstmhh0w_sigma = torch.randn_like(net.lstm.weight_hh_l0)
	lstmhh0w_mu_param = pyro.param("lstmhh0w_mu", lstmhh0w_mu)
	lstmhh0w_sigma_param = softplus(pyro.param("lstmhh0w_sigma", lstmhh0w_sigma))
	lstmhh0w_prior = Normal(loc=lstmhh0w_mu_param, scale=lstmhh0w_sigma_param)
	# lstm layer 1 hidden-hidden bias distribution priors
	lstmhh0b_mu = torch.randn_like(net.lstm.bias_hh_l0)
	lstmhh0b_sigma = torch.randn_like(net.lstm.bias_hh_l0)
	lstmhh0b_mu_param = pyro.param("lstmhh0b_mu", lstmhh0b_mu)
	lstmhh0b_sigma_param = softplus(pyro.param("lstmhh0b_sigma", lstmhh0b_sigma))
	lstmhh0b_prior = Normal(loc=lstmhh0b_mu_param, scale=lstmhh0b_sigma_param)
	# lstm layer 2 input-hidden weight distribution priors
	lstmih1w_mu = torch.randn_like(net.lstm.weight_ih_l1)
	lstmih1w_sigma = torch.randn_like(net.lstm.weight_ih_l1)
	lstmih1w_mu_param = pyro.param("lstmih1w_mu", lstmih1w_mu)
	lstmih1w_mu_param  = softplus(pyro.param("lstmih1w_sigma", lstmih1w_sigma))
	lstmih1w_prior = Normal(loc=lstmih1w_mu_param , scale=lstmih1w_mu_param )
	# lstm layer 2 input-hidden bias distribution priors
	lstmih1b_mu = torch.randn_like(net.lstm.bias_ih_l1)
	lstmih1b_sigma = torch.randn_like(net.lstm.bias_ih_l1)
	lstmih1b_mu_param = pyro.param("lstmih1b_mu", lstmih1b_mu)
	lstmih1b_sigma_param = softplus(pyro.param("lstmih1b_sigma", lstmih1b_sigma))
	lstmih1b_prior = Normal(loc=lstmih1b_mu_param, scale=lstmih1b_sigma_param)
	# lstm layer 2 hidden-hidden weight distribution priors
	lstmhh1w_mu = torch.randn_like(net.lstm.weight_hh_l1)
	lstmhh1w_sigma = torch.randn_like(net.lstm.weight_hh_l1)
	lstmhh1w_mu_param = pyro.param("lstmhh1w_mu", lstmhh1w_mu)
	lstmhh1w_sigma_param = softplus(pyro.param("lstmhh1w_sigma", lstmhh1w_sigma))
	lstmhh1w_prior = Normal(loc=lstmhh1w_mu_param, scale=lstmhh1w_sigma_param)
	# lstm layer 2 hidden-hidden bias distribution priors
	lstmhh1b_mu = torch.randn_like(net.lstm.bias_hh_l1)
	lstmhh1b_sigma = torch.randn_like(net.lstm.bias_hh_l1)
	lstmhh1b_mu_param = pyro.param("lstmhh1b_mu", lstmhh1b_mu)
	lstmhh1b_sigma_param = softplus(pyro.param("lstmhh1b_sigma", lstmhh1b_sigma))
	lstmhh1b_prior = Normal(loc=lstmhh1b_mu_param, scale=lstmhh1b_sigma_param)
	# first fully connected layer weight distribution priors
	fc1w_mu = torch.randn_like(net.fc1.weight)
	fc1w_sigma = torch.randn_like(net.fc1.weight)
	fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
	fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
	fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
	# first fully connected layer bias distribution priors
	fc1b_mu = torch.randn_like(net.fc1.bias)
	fc1b_sigma = torch.randn_like(net.fc1.bias)
	fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
	fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
	fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
	# second fully connected layer weight distribution priors
	fc2w_mu = torch.randn_like(net.fc2.weight)
	fc2w_sigma = torch.randn_like(net.fc2.weight)
	fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
	fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
	fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
	# Output layer bias distribution priors
	fc2b_mu = torch.randn_like(net.fc2.bias)
	fc2b_sigma = torch.randn_like(net.fc2.bias)
	fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
	fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
	fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)

	priors = {'embedding.weight': embedding_prior, 'lstm.weight_ih_l0': lstmih0w_prior, 'lstm.bias_ih_l0': lstmih0b_prior,
	'lstm.weight_hh_l0': lstmhh0w_prior, 'lstm.bias_hh_l0': lstmhh0b_prior, 'lstm.weight_ih_l0': lstmih0w_prior,
	'lstm.bias_ih_l0': lstmih0b_prior,'lstm.weight_hh_l0': lstmhh0w_prior, 'lstm.bias_hh_l0': lstmhh0b_prior,
	'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
	    
	lifted_module = pyro.random_module("module", net, priors)
	    
	return lifted_module()

def predict(num_samples, batch, tag):
	softmax = nn.Softmax(dim=1)
	net.eval()
	hidden = net.init_hidden(BATCH_SIZE)
	sampled_models = [guide(None, None, None, None) for _ in range(num_samples)]
	probs = []
	for model in sampled_models:
	    #for j in range(batch.size()[1]): 
	    	#output, hidden = model(batch[:,j], tag, hidden)
	    output, hidden = model(batch, tag, hidden)	
	    probs.append(softmax(output.data))
	mean = torch.mean(torch.stack(probs), 0).cpu()
	var = torch.var(torch.stack(probs), 0).cpu()
	#lower = torch.kthvalue(torch.stack(probs), 2, 0).values.cpu()
	#upper = torch.kthvalue(torch.stack(probs), 24, 0).values.cpu()
	preds = np.argmax(mean.numpy(), axis=1)
	for i in range(BATCH_SIZE):
		if var[i][preds[i]] > .23:
		#if lower[i][preds[i]] < .5 and upper[i][preds[i]] > .5:
			preds[i] = 99
	net.train()
	return preds 


if __name__ == '__main__':
	device = "cuda" #make sure on GPU
	BATCH_SIZE = 250
	NUM_EPOCHS = 50

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
	tweets_brand = data_dict['tweets_brand']
	hashtags_brand = data_dict['hashtags_brand']
	labels_brand = data_dict['labels_brand']

	#initialize language, adds each tweet to the language
	twitter_lang = Lang()
	for tweet in tweets_train: #make language from all tweets
		twitter_lang.add_tweet(tweet)
	for tweet in tweets_test:
		twitter_lang.add_tweet(tweet)
	for tweet in tweets_brand:
		twitter_lang.add_tweet(tweet)

	pad_index = twitter_lang.word2index['<pad>']

	#initialize network
	net = Net(input_size = twitter_lang.n_words, 
				hidden_size = 100, 
				embedding_size = 10,
				padding_idx =  pad_index).cuda() #hidden size is arbitrary for now
	criterion = nn.CrossEntropyLoss() #maybe experiment with different loss. NLLL?
	optimizer = Adam({"lr": 0.001}) #weight decay??

	#lists of tensors basically
	tweets_train = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_train], 0)
	tweets_test = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_test], 0)
	tweets_brand = torch.stack([tensor_from_text(twitter_lang, tweet).cuda() for tweet in tweets_brand], 0)

	labels_train = torch.stack([torch.tensor([label]).cuda() for label in labels_train], 0)
	labels_test = torch.stack([torch.tensor([label]).cuda() for label in labels_test], 0)
	labels_brand = torch.stack([torch.tensor([label]).cuda() for label in labels_brand], 0)

	hashtags_train = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_train], 0) #just using number of hashtags used in tweet
	hashtags_test = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_test], 0)
	hashtags_brand = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_brand], 0)

	#if you want to pretrain
	pretrained_dict = torch.load("lstm.pth") 
	net.train()

	#model_dict = net.state_dict(pretrained_dict)

	# stochastic variational inference
	svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

	acc_matrix = np.zeros((NUM_EPOCHS, 2))
	fail_matrix = np.zeros((NUM_EPOCHS, 3))

	#now we're training
	print("training on tweets...")
	for epoch in range(NUM_EPOCHS): 
		permutation = torch.randperm(tweets_train.size()[0]) #shuffle batches
		iters = 1

		for i in range(0,tweets_train.size()[0] - tweets_train.size()[0]%BATCH_SIZE, BATCH_SIZE):
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			hidden = net.init_hidden(BATCH_SIZE)
			#for j in range(batch_tweets.size()[1]): 
				#loss = svi.step(batch_tweets[:,j], batch_hashtags, hidden, batch_labels)
			loss = svi.step(batch_tweets, batch_hashtags, hidden, batch_labels)


		#eval training data
		correct = 0
		total = 0
		fail = 0
		for i in range(0,tweets_train.size()[0] - tweets_train.size()[0]%BATCH_SIZE, BATCH_SIZE):
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			predicted = predict(25, batch_tweets, batch_hashtags)
			num_correct = np.sum(predicted == batch_labels.squeeze().data.cpu().numpy())
			num_fail = np.sum(predicted == 99)
			total += batch_labels.size(0) - num_fail
			correct += num_correct
			fail += num_fail
		train_acc = correct / total
		train_fail = fail / (tweets_train.size()[0] - tweets_train.size()[0]%BATCH_SIZE)
		acc_matrix[epoch, 0] = train_acc
		fail_matrix[epoch, 0] = train_fail


		#eval test data
		correct = 0
		total = 0
		fail = 0
		for i in range(0,tweets_test.size()[0] - tweets_test.size()[0]%BATCH_SIZE, BATCH_SIZE):
			batch_tweets, batch_hashtags, batch_labels = tweets_test[i:i+BATCH_SIZE], hashtags_test[i:i+BATCH_SIZE], labels_test[i:i+BATCH_SIZE]
			predicted = predict(25, batch_tweets, batch_hashtags)
			num_correct = np.sum(predicted == batch_labels.squeeze().data.cpu().numpy())
			num_fail = np.sum(predicted == 99)
			total += batch_labels.size(0) - num_fail
			correct += num_correct
			fail += num_fail
		test_acc = correct / total
		test_fail = fail / (tweets_test.size()[0] - tweets_test.size()[0]%BATCH_SIZE)
		acc_matrix[epoch, 1] = test_acc
		fail_matrix[epoch, 1] = test_fail

		total = 0
		fail = 0
		for i in range(0,tweets_brand.size()[0] - tweets_brand.size()[0]%BATCH_SIZE, BATCH_SIZE):
			batch_tweets, batch_hashtags, batch_labels = tweets_brand[i:i+BATCH_SIZE], hashtags_brand[i:i+BATCH_SIZE], labels_brand[i:i+BATCH_SIZE]
			predicted = predict(25, batch_tweets, batch_hashtags)
			total += batch_labels.size(0)
			num_fail = np.sum(predicted == 99)
			fail += num_fail
		brand_fail = fail / total
		fail_matrix[epoch, 2] = brand_fail

		print('EPOCH: %d' %
              (epoch+1))
		print('train_acc: %.5f prop_fail: %.5f' %
			  (train_acc, train_fail))
		print('test_acc: %.5f prop_fail: %.5f' %
			  (test_acc, test_fail))
		print('brand prop_fail: %.5f' %
			  (brand_fail))
	#torch.save(net.state_dict(), 'mytraining.pth') #save state dictionary
	np.savetxt("blstm_acc.csv", acc_matrix, delimiter=",")
	np.savetxt("blstm_fail.csv", fail_matrix, delimiter=",")

