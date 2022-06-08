'''
Model code from https://github.com/caitlinsmith14/gestnet
See the paper https://pages.jh.edu/csmit372/pdf/smithohara_scil2021_paper.pdf
Edits by Hilly Steinmetz
'''
import glob
from math import ceil
from typing import Callable
import matplotlib.pyplot as plt
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm
import warnings
import csv
from functools import reduce
from typing import *
from math import ceil, floor


# outline for testing various models:
# input = [la, tb, tc]
# output_size = len(input) ... 2-3
# dataset can stay the same as long as the right articulators are pulled up 
# init decoder with that output size
# also need to change the eval and train functions to accommodate multiple types of articulators

# (remember want to average across several models!)

##############
# PREPROCESS #
##############

class LinearTransform:
	'''
	Scales the data for each gesture to ensure that the network learns equallly from each
	'''
	def __init__(self, class_labels: str) -> None:
		min_str = lambda x: min([int(a) for a in x.split(",")]) if isinstance(x, str) else x
		min_str_list = lambda x, y: min(min_str(x), min_str(y))
		self.lowest_num = reduce(min_str_list, class_labels)
		max_str = lambda x: max([int(a) for a in x.split(",")]) if isinstance(x, str) else x
		max_str_list = lambda x, y: max(max_str(x), max_str(y))
		self.highest_num = reduce(max_str_list, class_labels)
		self.range = float(self.highest_num - self.lowest_num)

	def list_transform(self, x: str, operator: Callable, y: float):
		r = list(map(lambda x: operator(float(x), y), [float(a) for a in x.split(",")]))
		return r

	def linear_transform_list(self, matrix: str):
		div = lambda x, y: x / y
		r = self.list_transform(matrix, div, self.range)
		return r

	def linear_transform(self, matrix: torch.Tensor):
		return torch.divide(matrix, self.range)

	def linear_transform_back(self, matrix: List[float]):
		r = [x * self.range for x in matrix]
		return r

###########
# DATASET #
###########

class Dataset:  # Dataset object class utilizing Torchtext

	def __init__(self, path, batch_size=1):
		print(f"{path}")
		self.batch_size = batch_size
		self.input_field, self.output_field_la, self.output_field_tb, self.output_field_tc, self.data, self.data_iter = self.process_data(path)
		self.word2trialnum = self.make_trial_lookup(path)
		self.seg2ind = self.input_field.vocab.stoi  # from segment to torchtext vocab index

	def process_data(self, path):  # create Dataset object from tab-delimited text file

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		make_list = lambda b: [item for item in b.split(',')]
		# make_float = lambda b: [float(item) for item in b.split(',')]

		with open(path, 'r') as file:
			tsv_reader = csv.reader(file, delimiter='\t', quotechar='"')
			next(tsv_reader)
			vocabulary, la_outs, tb_outs, tc_outs = [], [], [], []
			for row in tsv_reader:
				vocabulary.append(row[0])
				la_outs.append(row[5])
				tb_outs.append(row[6])
				tc_outs.append(row[7])

		self.vocabulary = vocabulary

		self.linear_tr_la = LinearTransform(la_outs)
		self.linear_tr_tb = LinearTransform(tb_outs)
		self.linear_tr_tc = LinearTransform(tc_outs)

		linear_tr_la = lambda x: self.linear_tr_la.linear_transform_list(x)
		linear_tr_tb = lambda x: self.linear_tr_tb.linear_transform_list(x)
		linear_tr_tc = lambda x: self.linear_tr_tc.linear_transform_list(x)

		# for easier retrieval later on
		self.articulator_ref = {'la_output':self.linear_tr_la, 'tb_output':self.linear_tr_tb, 'tc_output':self.linear_tr_tc}

		input_field = Field(sequential=True, use_vocab=True, tokenize=make_list)  # morpheme segment format
		output_field_la = Field(sequential=True, use_vocab=False, tokenize=linear_tr_la, pad_token=0, dtype=torch.float)  # lip trajectory outputs
		output_field_tb = Field(sequential=True, use_vocab=False, tokenize=linear_tr_tb, pad_token=0, dtype=torch.float)   # tb trajectory outputs
		output_field_tc = Field(sequential=True, use_vocab=False, tokenize=linear_tr_tc, pad_token=0, dtype=torch.float)   # tb trajectory outputs

		datafields = [('underlying', None), ('surface', None), ('root_indices', None), ('suffix_indices', None),
					  ('word_indices', input_field), ('la_output', output_field_la), ('tb_output', output_field_tb), ('tc_output', output_field_tc)]

		data = TabularDataset(path=path, format='tsv', skip_header=True, fields=datafields)

		# test_data = ["t-i","t-a","t-e","t-o","t-u","t-y","t-ø","t-ɯ","it-H","at-H","et-H","ot-H","ut-H","yt-H","øt-H","ɯt-H","it-L","at-L","et-L","ot-L","ut-L","yt-L","øt-L","ɯt-L"]

		input_field.build_vocab(data, min_freq=1)
		data_iter = BucketIterator(data,
								   batch_size=self.batch_size,
								   sort_within_batch=False,
								   repeat=False,
								   device=device)

		return input_field, output_field_la, output_field_tb, output_field_tc, data, data_iter

	def make_trial_lookup(self, path):  # create lookup dictionary for use by make_trial method
		with open(path, 'r') as file:
			word2trialnum = {}
			for x, line in enumerate(file, start=-1):
				word2trialnum[line.split()[0]] = x
		return word2trialnum

	def make_trial(self, word):  # get target outputs for individual word for use by Seq2Seq's evaluate_word method

		trialnum = self.word2trialnum[word]

		source = self.data.examples[trialnum].word_indices
		la_target = self.data.examples[trialnum].la_output
		tb_target = self.data.examples[trialnum].tb_output
		tc_target = self.data.examples[trialnum].tc_output

		source_list = []

		for seg in source:
			source_list.append(self.seg2ind[seg])

		source_tensor = torch.tensor(source_list, dtype=torch.long).view(-1, 1)

		la_target_tensor = torch.tensor(la_target, dtype=torch.float).view(-1, 1)
		tb_target_tensor = torch.tensor(tb_target, dtype=torch.float).view(-1, 1)
		tc_target_tensor = torch.tensor(tc_target, dtype=torch.float).view(-1, 1)

		return source_tensor, la_target_tensor, tb_target_tensor, tc_target_tensor

	def transform_back(self, articulator: str, values: List[int]):
		'''Get the linear transform of an example back from an articulator'''
		linear_tr = self.articulator_ref[articulator]
		return linear_tr.linear_transform_back(values)


###########
# ENCODER #
###########

class Encoder(nn.Module):

	def __init__(self,
				 vocab_size,  # size of vector representing each segment in vocabulary (created by Dataset)
				 seg_embed_size,  # size of segment embedding
				 hidden_size):  # size of encoder hidden layer
		super(Encoder, self).__init__()

		self.params = (vocab_size, seg_embed_size, hidden_size)

		self.embedding = nn.Embedding(vocab_size, seg_embed_size)  # embedding dictionary for each segment
		self.rnn = nn.RNN(seg_embed_size, hidden_size)  # RNN hidden layer

	def forward(self, input_seq):
		embedded_seq = self.embedding(input_seq)
		output_seq, last_hidden = self.rnn(embedded_seq)
		return output_seq, last_hidden, embedded_seq


#############################
# ENCODER-DECODER ATTENTION #
#############################

class EncoderDecoderAttn(nn.Module):  # attention mechanism between encoder and decoder hidden states

	def __init__(self,
				 encoder_size,  # size of encoder hidden layer
				 decoder_size,  # size of decoder hidden layer
				 attn_size):  # size of attention vector
		super(EncoderDecoderAttn, self).__init__()  # always call this

		self.params = (encoder_size, decoder_size, attn_size)

		self.linear = nn.Linear(encoder_size+decoder_size, attn_size)  # linear layer

	def forward(self, decoder_hidden, encoder_outputs):

		decoder_hidden = decoder_hidden.squeeze(0)
		input_seq_length = encoder_outputs.shape[0]

		repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, input_seq_length, 1)
		encoder_outputs = encoder_outputs.permute(1, 0, 2)
		attn = torch.tanh(self.linear(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))

		attn_sum = torch.sum(attn, dim=2)

		attn_softmax = F.softmax(attn_sum, dim=1).unsqueeze(1)
		attended_encoder_outputs = torch.bmm(attn_softmax, encoder_outputs).squeeze(1)

		encoder_norms = encoder_outputs.norm(dim=2)
		attn_map = attn_softmax.squeeze(1) * encoder_norms

		return attended_encoder_outputs, attn_map


###########
# DECODER #
###########

class Decoder(nn.Module):

	def __init__(self,
				 hidden_size,  # size of hidden layer for both encoder and decoder
				 attn,         # encoder-decoder attention mechanism
				 output_size): # the number of articulators (lip and tongue body)
		super(Decoder, self).__init__()

		self.params = hidden_size

		self.output_size = output_size  # number of articulators 

		self.attn = attn  # encoder-decoder attention mechamism
		self.rnn = nn.RNN(self.output_size+self.attn.params[0], hidden_size)  # RNN hidden layer
		self.linear = nn.Linear(hidden_size, self.output_size)  # linear layer

	def forward(self, input_tok, hidden, encoder_outputs):
		input_tok = input_tok.float()

		attended, attn_map = self.attn(hidden, encoder_outputs)
		rnn_input = torch.cat((input_tok, attended), dim=1).unsqueeze(0)

		output, hidden = self.rnn(rnn_input, hidden)
		output = self.linear(output.squeeze(0))

		return output, hidden, attn_map


##############################
# SEQUENCE TO SEQUENCE MODEL #
##############################

class Seq2Seq(nn.Module):  # Combine encoder and decoder into sequence-to-sequence model
	def __init__(self,
				 training_data=None,  # training data (Dataset class object)
				 load='',  # path to file for loading previously trained model
				 seg_embed_size=64,  # size of segment embedding
				 hidden_size=64,  # size of hidden layer
				 attn_size=64,  # size of encoder-decoder attention vector
				 articulators=['la_output', 'tb_output'], # the articulators used to train
				 optimizer='adam',  # what type of optimizer (Adam or SGD)
				 learning_rate=.0001):  # learning rate of the model
		super(Seq2Seq, self).__init__()

		# Seq2Seq Parameters

		self.init_input_tok = nn.Parameter(torch.rand(1, len(articulators)))  # initialize first decoder input (learnable)

		# Hyperparameters / Device Settings

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.loss_function = nn.MSELoss(reduction='mean')

		# Load a trained model and its subcomponents

		if load:
			self.path = re.sub('_[0-9]+.pt', '_', load)
			checkpoint = torch.load(load)

			self.articulators = checkpoint['articulators']
			assert articulators == self.articulators

			self.encoder = Encoder(vocab_size=checkpoint['encoder_params'][0],
								   seg_embed_size=checkpoint['encoder_params'][1],
								   hidden_size=checkpoint['encoder_params'][2])

			attn = EncoderDecoderAttn(encoder_size=checkpoint['attn_params'][0],
									  decoder_size=checkpoint['attn_params'][1],
									  attn_size=checkpoint['attn_params'][2])

			self.decoder = Decoder(hidden_size=checkpoint['decoder_params'],
								   attn=attn, output_size=len(self.articulators))

			self.loss_list = checkpoint['loss_list']

			self.load_state_dict(checkpoint['seq2seq_state_dict'])

			if checkpoint['optimizer_type'] == 'SGD':
				self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
			elif checkpoint['optimizer_type'] == 'Adam':
				self.optimizer = optim.Adam(self.parameters())
			else:
				print('Optimizer not loaded! Try again.')
				return

			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		else:  # Initialize a new model and its subcomponents
			if not training_data:
				print('Required input: training_data (Dataset class object). Try again!')
				return

			self.path = None

			# set the articulators as an attribute (to help with saving and loading)
			self.articulators = articulators

			self.encoder = Encoder(vocab_size=len(training_data.input_field.vocab),
								   seg_embed_size=seg_embed_size,
								   hidden_size=hidden_size)

			attn = EncoderDecoderAttn(encoder_size=hidden_size,
									  decoder_size=hidden_size,
									  attn_size=attn_size)

			self.decoder = Decoder(hidden_size=hidden_size,
								   attn=attn, output_size=len(self.articulators))

			self.loss_list = []

			for name, param in self.named_parameters():
				nn.init.uniform_(param.data, -0.08, 0.08)

			if optimizer == 'adam':
				self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
			elif optimizer == 'sgd':
				self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
			else:
				'No such optimizer! Try again.'
				return

	def forward(self, input_seq, target_seq):
		input_length = input_seq.shape[0]
		target_length = target_seq.shape[0]
		target_output_size = self.decoder.output_size

		output_seq = torch.zeros(target_length, target_output_size, dtype=float).to(self.device)
		attn_map_seq = torch.zeros(target_length, input_length, dtype=float).to(self.device)

		encoder_outputs, hidden, embeddings = self.encoder(input_seq)

		input_tok = self.init_input_tok

		for t in range(target_length):
			output_tok, hidden, attn_map = self.decoder(input_tok,
														hidden,
														encoder_outputs)

			output_seq[t] = output_tok
			attn_map_seq[t] = attn_map

			input_tok = output_tok

		return output_seq, attn_map_seq

	def train_model(self, training_data, n_epochs=1):  # Train the model on the provided dataset

		self.train()

		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for e in tqdm(range(n_epochs), leave=False, display=False):

				early_stop_loss = 0
				previous_loss = 1000

				for i, batch in enumerate(training_data.data_iter):
					self.zero_grad()
					source = batch.word_indices
					target_values = []
					for art in self.articulators:
						target_values.append(getattr(batch, art))
					target = torch.cat(tuple(target_values), axis=1)
					mask = torch.where(target != 0, 1., 0.)
					predicted, enc_dec_attn_seq = self(source, target)
					predicted_masked = mask * predicted

					loss = self.loss_function(predicted_masked.float(), target.float())
					loss.backward()
					self.optimizer.step()
				
				avg_loss = self.evaluate_model(training_data, verbose=False)

				if avg_loss > previous_loss:
					early_stop_loss += 1
					if early_stop_loss > 4:
						print(f"EARLY STOPPING AT EPOCH {e}")
						break
				else:
					early_stop_loss = 0
				previous_loss = avg_loss

				self.loss_list.append(self.evaluate_model(training_data, verbose=False))

	def evaluate_model(self, training_data, verbose=True):  # Evaluate the model's performance on the dataset
		self.eval()
		epoch_loss = 0

		with warnings.catch_warnings():
			warnings.simplefilter('ignore')

			with torch.no_grad():

				for i, batch in enumerate(training_data.data_iter):
					source = batch.word_indices
					target_values = []
					for art in self.articulators:
						target_values.append(getattr(batch, art)) # get the articulators relevant to the output
					target = torch.cat(tuple(target_values), axis=1)
					predicted, _ = self(source, target)

					loss = self.loss_function(predicted.float(), target.float())
					epoch_loss += loss.item()

			average_loss = epoch_loss / len(training_data.data_iter)

			if verbose:
				# changed to cross entropy loss
				print(f'Average loss per word this epoch:')

			return average_loss

	def plot_loss(self):  # Plot the model's average trial loss per epoch
		plt.plot(self.loss_list, '-')
		plt.title('Average Trial Loss Per Epoch')
		plt.ylabel('MSE Loss')
		plt.xlabel('Epoch')

	def evaluate_word(self, training_data, word, show_target=True):  # Evaluate the model's performance on a single word
		self.eval()

		trial = training_data.make_trial(word)

		with torch.no_grad():
			source = trial[0]
			target_values = []
			target_dict = {'la_output':1, 'tb_output':2, 'tc_output': 3}
			for art in self.articulators:
				target_values.append(trial[target_dict[art]]) # get the articulators relevant to the output
			target = torch.cat(tuple(target_values), 1)

			predicted, enc_dec_attn_seq = self(source, target)
			print(f'Target output:\n{target}')
			print(f'Predicted output:\n{predicted}')
			print(f'Encoder Decoder Attention:\n{enc_dec_attn_seq}')

		# transform the predictions back
		predicted_values = {}
		for i, art in enumerate(self.articulators):
			predicted_values[art] = training_data.transform_back(art, predicted[:, i])
	
		# transform the targets back
		target_values = {}
		for i, art in enumerate(self.articulators):
			target_values[art] = training_data.transform_back(art, target[:, i])

		# set the titles of the plot for each articulator
		tract_labels = {
			'la_output': "Lip Aperture (Vowel Rounding)", 
			'tb_output': 'Tongue Body Closure (Vowel Height)', 
			'tc_output': 'Palatal Constriction (Vowel Backness)'
			}

		# plot the results
		for art in self.articulators:
			figure_outputs, (art_plot) = plt.subplots(1)

			figure_outputs.suptitle('Predicted Tract Variable Trajectories')

			predicted_art = predicted_values[art]
			target_art = target_values[art]

			art_label = tract_labels[art]

			# Articulator Trajectory Subplot

			art_plot.plot(predicted_art, label='Predicted')
			if show_target:
				art_plot.plot(target_art, label='Target')
			if target_art[-4:] == [0,0,0,0]:
				art_plot.set_xlim(0,5)
			art_plot.set_title(art_label)
			art_plot.set_ylabel('Constriction Degree')
			
			# set the limits of each chart
			max_y = max(max(predicted_art), max(target_art))
			min_y = min(min(predicted_art), min(target_art))
			y_max = ceil(max_y + (0.05 * max_y))
			y_min = floor(min_y - (0.05 * abs(min_y)))
			art_plot.set_ylim(y_max, y_min)

			art_plot.legend()

		# Plot Encoder-Decoder Attention

		heatmap_attn, ax = plt.subplots()
		heatmap_attn.suptitle('Encoder-Decoder Attention')
		im = ax.imshow(enc_dec_attn_seq.permute(1, 0), cmap='gray')

		ax.set_xticks([x for x in range(enc_dec_attn_seq.shape[0])])
		ax.set_xticklabels([x+1 for x in range(enc_dec_attn_seq.shape[0])])
		ax.set_xlabel('Decoder Time Point')

		ax.set_yticks([x for x in range(len(re.sub('-', '', word)))])
		ax.set_yticklabels(list(re.sub('-', '', word)))
		ax.set_ylabel('Input')

		plt.show()

	def save(self):  # Save model to 'saved_models'

		save_dict = {'encoder_params': self.encoder.params,
					 'attn_params': self.decoder.attn.params,
					 'decoder_params': self.decoder.params,
					 'articulators': self.articulators, 
					 'seq2seq_state_dict': self.state_dict(),
					 'optimizer_type': str(self.optimizer)[0:4].strip(),
					 'optimizer_state_dict': self.optimizer.state_dict(),
					 'loss_list': self.loss_list}

		if not os.path.isdir('saved_models'):
			os.mkdir('saved_models')

		if self.path is None:
			model_num = 1
			arts = "_".join([s[0:2] for s in self.articulators])
			while glob.glob(os.path.join('saved_models', f'gestnet_{arts}_{model_num}_*.pt')):
				model_num += 1

			self.path = os.path.join('saved_models', f'gestnet_{arts}_{model_num}_')
		else:
			model_num = self.path.split('_')[-2]

		saveas = f'{self.path}{str(len(self.loss_list))}.pt'

		torch.save(save_dict, saveas)
		print(f'Model saved as gestnet_{arts}_{model_num}_{str(len(self.loss_list))} in directory saved_models.')

	def count_params(self):  # Count trainable model parameters
		params = sum(p.numel() for p in self.parameters() if p.requires_grad)
		print('The model has ' + str(params) + ' trainable parameters.')

