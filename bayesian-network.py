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
        self.gru = nn.GRU(input_size = embedding_size, 
        					hidden_size = hidden_size,
        					num_layers=1) #trying different nuymber of layers
        self.fc1 = nn.Linear(hidden_size + 1, 20) #try different number of hidden units?
        self.fc2 = nn.Linear(20, 3)
        self.leakyrelu = nn.LeakyReLU()

    #forward pass
    def forward(self, batch, tag, hidden):
    	embed = self.embedding(batch).view(1, BATCH_SIZE, -1) #embedding of vector
    	output = embed
    	self.gru.flatten_parameters()
    	output, hidden = self.gru(output, hidden) #GRU layer
    	output = torch.cat((output.squeeze(), tag), dim = 1) #concatinate hashtag count
    	output = self.leakyrelu(self.fc1(output)) #one fully connected layer
    	output = self.fc2(output) #second FC layer
    	return output, hidden

    #initializes the hidden state for a new tensor input
    #maybe we can learn initial states?
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device) #change first dimension for number of hidden layers of gru

# define priors
def model(batch, tag, hidden, label):
	log_softmax = nn.LogSoftmax(dim=1)
	    
	embedding_prior = Normal(loc=pretrained_dict['embedding.weight'], scale=torch.ones_like(net.embedding.weight))

	gruihw_prior = Normal(loc=pretrained_dict['gru.weight_ih_l0'], scale=torch.ones_like(net.gru.weight_ih_l0))
	gruihb_prior = Normal(loc=pretrained_dict['gru.bias_ih_l0'], scale=torch.ones_like(net.gru.bias_ih_l0))
	gruhhw_prior = Normal(loc=pretrained_dict['gru.weight_hh_l0'], scale=torch.ones_like(net.gru.weight_hh_l0))
	gruhhb_prior = Normal(loc=pretrained_dict['gru.bias_hh_l0'], scale=torch.ones_like(net.gru.bias_hh_l0))

	fc1w_prior = Normal(loc=pretrained_dict['fc1.weight'], scale=torch.ones_like(net.fc1.weight))
	fc1b_prior = Normal(loc=pretrained_dict['fc1.bias'], scale=torch.ones_like(net.fc1.bias))
	    
	fc2w_prior = Normal(loc=pretrained_dict['fc2.weight'], scale=torch.ones_like(net.fc2.weight))
	fc2b_prior = Normal(loc=pretrained_dict['fc2.bias'], scale=torch.ones_like(net.fc2.bias))
	    
	priors = {'embedding.weight': embedding_prior, 'gru.weight_ih_l0': gruihw_prior, 'gru.bias_ih_l0': gruihb_prior,
	'gru.weight_hh_l0': gruhhw_prior, 'gru.bias_hh_l0': gruhhb_prior, 
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
	# gru input-hidden weight distribution priors
	gruihw_mu = torch.randn_like(net.gru.weight_ih_l0)
	gruihw_sigma = torch.randn_like(net.gru.weight_ih_l0)
	gruihw_mu_param = pyro.param("gruihw_mu", gruihw_mu)
	gruihw_sigma_param = softplus(pyro.param("gruihw_sigma", gruihw_sigma))
	gruihw_prior = Normal(loc=gruihw_mu_param, scale=gruihw_sigma_param)
	# gru input-hidden bias distribution priors
	gruihb_mu = torch.randn_like(net.gru.bias_ih_l0)
	gruihb_sigma = torch.randn_like(net.gru.bias_ih_l0)
	gruihb_mu_param = pyro.param("gruihb_mu", gruihb_mu)
	gruihb_sigma_param = softplus(pyro.param("gruihb_sigma", gruihb_sigma))
	gruihb_prior = Normal(loc=gruihb_mu_param, scale=gruihb_sigma_param)
	# gru hidden-hidden weight distribution priors
	gruhhw_mu = torch.randn_like(net.gru.weight_hh_l0)
	gruhhw_sigma = torch.randn_like(net.gru.weight_hh_l0)
	gruhhw_mu_param = pyro.param("gruhhw_mu", gruhhw_mu)
	gruhhw_sigma_param = softplus(pyro.param("gruhhw_sigma", gruhhw_sigma))
	gruhhw_prior = Normal(loc=gruhhw_mu_param, scale=gruhhw_sigma_param)
	# gru hidden-hidden bias distribution priors
	gruhhb_mu = torch.randn_like(net.gru.bias_hh_l0)
	gruhhb_sigma = torch.randn_like(net.gru.bias_hh_l0)
	gruhhb_mu_param = pyro.param("gruhhb_mu", gruhhb_mu)
	gruhhb_sigma_param = softplus(pyro.param("gruhhb_sigma", gruhhb_sigma))
	gruhhb_prior = Normal(loc=gruhhb_mu_param, scale=gruhhb_sigma_param)
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

	priors = {'embedding.weight': embedding_prior, 'gru.weight_ih_l0': gruihw_prior, 'gru.bias_ih_l0': gruihb_prior,
	'gru.weight_hh_l0': gruhhw_prior, 'gru.bias_hh_l0': gruhhb_prior, 
	'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
	    
	lifted_module = pyro.random_module("module", net, priors)
	    
	return lifted_module()

def predict(num_samples, batch, tag):
	net.eval()
	hidden = net.init_hidden(BATCH_SIZE)
	sampled_models = [guide(None, None, None, None) for _ in range(num_samples)]
	yhats = []
	for model in sampled_models:
	    for j in range(batch.size()[1]): 
	    	output, hidden = model(batch[:,j], tag, hidden)
	    yhats.append(output.data)
	mean = torch.mean(torch.stack(yhats), 0).cpu()
	net.train()
	return np.argmax(mean.numpy(), axis=1) 


if __name__ == '__main__':
	device = "cuda" #make sure on GPU
	BATCH_SIZE = 250
	NUM_EPOCHS = 10

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

	labels_train = torch.stack([torch.tensor([label]).cuda() for label in labels_train], 0)
	labels_test = torch.stack([torch.tensor([label]).cuda() for label in labels_test], 0)

	hashtags_train = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_train], 0) #just using number of hashtags used in tweet
	hashtags_test = torch.stack([torch.tensor([len(hashtag)]).cuda().float() for hashtag in hashtags_test], 0)

	#if you want to pretrain
	pretrained_dict = torch.load("mytraining.pth") 
	net.train()

	#model_dict = net.state_dict(pretrained_dict)

	svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

	#now we're training
	print("training on tweets...")
	for epoch in range(NUM_EPOCHS): 
		permutation = torch.randperm(tweets_train.size()[0]) #shuffle batches
		iters = 1

		for i in range(0,tweets_train.size()[0], BATCH_SIZE):
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			hidden = net.init_hidden(BATCH_SIZE)
			for j in range(batch_tweets.size()[1]): 
				loss = svi.step(batch_tweets[:,j], batch_hashtags, hidden, batch_labels)


		#eval training data
		correct = 0
		total = 0
		for i in range(0,tweets_train.size()[0], BATCH_SIZE):
			indices = permutation[i:i+BATCH_SIZE] #random batches
			batch_tweets, batch_hashtags, batch_labels = tweets_train[indices], hashtags_train[indices], labels_train[indices]
			predicted = predict(10, batch_tweets, batch_hashtags)
			num_correct = np.sum(predicted == batch_labels.cpu().numpy())
			total += batch_labels.size(0)
			correct += num_correct
		train_acc = correct / total


		#eval test data
		correct = 0
		total = 0
		for i in range(0,tweets_test.size()[0] - tweets_test.size()[0]%BATCH_SIZE, BATCH_SIZE):
			batch_tweets, batch_hashtags, batch_labels = tweets_test[i:i+BATCH_SIZE], hashtags_test[i:i+BATCH_SIZE], labels_test[i:i+BATCH_SIZE]
			predicted = predict(10, batch_tweets, batch_hashtags)
			num_correct = np.sum(predicted == batch_labels.cpu().numpy())
			total += batch_labels.size(0)
			correct += num_correct
		test_acc = correct / total

		print('EPOCH: %d train_acc: %.5f test_acc: %.5f' %
              (epoch+1, train_acc, test_acc))
	#torch.save(net.state_dict(), 'mytraining.pth') #save state dictionary
