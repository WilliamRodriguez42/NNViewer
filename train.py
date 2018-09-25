import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import codecs
import json
import logging
import os
import sys

import numpy as np
import char_rnn_model
from char_rnn_model import *
from six import iteritems
import string
from datetime import *
import time
import io

learning_rate = -1
file_read = False

def train(settings):
	global learning_rate, file_read

	learning_rate = -1
	file_read = False

	# Copy all arg variables to scope
	batch_size = settings['Batch Size']
	best_model = ''
	best_valid_ppl = np.Inf
	data_file = settings['Data File']
	dropout = settings['Dropout']
	embedding_size = 0
	encoding = 'utf-8'
	hidden_size = settings['Hidden Size']
	init_dir = settings['Initialize Directory']
	init_model = ''
	input_dropout = settings['Input Dropout']
	length = settings['Length']
	max_learning_rate = settings['Current Learning Rate']
	max_grad_norm = 5
	max_to_keep = 5
	min_learning_rate = settings['Final Learning Rate']
	model = settings['Model']
	n_save = settings['Save Frequency']
	num_epochs = settings['Number of Epochs']
	num_layers = settings['Number of Layers']
	num_unrollings = settings['Number of Unrollings']
	output_base_dir = settings['Output Directory']
	progress_freq = 100
	project_name = settings['Project Name']
	verbose = 0

	output_dir = os.path.join(output_base_dir, project_name)

	train_frac = settings['Train Fraction']
	valid_frac = settings['Valid Fraction']
	test_frac = 1 - valid_frac - train_frac

	save_path = os.path.join(output_dir, 'saved_models')
	best_path = os.path.join(output_dir, 'best_model')
	tb_log_dir = os.path.join(output_dir, 'tensorboard_log/')

	# Generate params
	params = {
			  'batch_size': batch_size,
			  'num_unrollings': num_unrollings,

			  'dropout':dropout,
			  'input_dropout': input_dropout,

			  'embedding_size': embedding_size,
			  'hidden_size': hidden_size,
			  'num_layers': num_layers,

			  'max_grad_norm': max_grad_norm,
			  'max_learning_rate': max_learning_rate,
			  'min_learning_rate': min_learning_rate,
			  'decay_rate' : 0,

			  'model': model,
			  'vocab_size' : 0
			  }

	if init_dir:
		with open(os.path.join(init_dir, 'result.json'), 'r') as f:
			result = json.load(f)
		fparams = result['params']
		init_model = result['latest_model']
		best_model = result['best_model']
		best_valid_ppl = result['best_valid_ppl']
		test_ppl = result['test_ppl']

		vocab_size = fparams['vocab_size']
		result['params'] = params

	else:
		init_model = ''
		best_model = ''
		best_valid_ppl = ''
		test_ppl = ''

		result = {
			'init_model' : init_model,
			'best_model' : best_model,
			'best_valid_ppl' : best_valid_ppl,
			'test_ppl' : test_ppl,
			'encoding' : encoding,
			'params' : params
		}

	# Create necessary directories.
	if init_dir:
		output_dir = init_dir
	else:
		for paths in [save_path, best_path, tb_log_dir]:
			os.makedirs(os.path.dirname(paths), exist_ok=True)

	# Set logging to stdout
	logging.basicConfig(stream=sys.stdout,
						format='%(asctime)s %(levelname)s:%(message)s',
						level=logging.INFO,
						datefmt='%I:%M:%S')

	logging.info('Settings:')
	for key, item in settings.items():
		logging.info(str(key) + ': ' + str(item))

	logging.info('Results:')
	for key, item in result.items():
		logging.info(str(key) + ': ' + str(item))

	logging.info('=' * 60)
	logging.info('All final and intermediate outputs will be stored in %s/' % output_dir)
	logging.info('=' * 60 + '\n')

	# Read and split data.
	logging.info('Reading data from: %s', data_file)

	with codecs.open(data_file, 'r', encoding=encoding) as f:
		text = f.read()
	file_read = True

	if length > 0:
		text = text[:length]

	logging.info('Number of characters: %s', len(text))

	logging.info('Creating train, valid, test split')
	train_size = int(train_frac * len(text))
	valid_size = int(valid_frac * len(text))
	test_size = len(text) - train_size - valid_size
	train_text = text[:train_size]
	valid_text = text[train_size:train_size + valid_size]
	test_text = text[train_size + valid_size:]

	# Calculate total number of steps (round up)
	steps_per_epoch = train_size // (batch_size * num_unrollings)
	if train_size % (batch_size * num_unrollings) != 0:
		steps_per_epoch += 1

	total_steps = round(steps_per_epoch * num_epochs + 0.5)
	percentOfLearningRate = min_learning_rate / max_learning_rate
	decay_rate = np.power(percentOfLearningRate, 1 / total_steps)
	params['decay_rate'] = decay_rate

	vocab_file = os.path.join(output_dir, 'vocab.json')
	if init_dir:
		vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(vocab_file, encoding)
	else:
		logging.info('Creating vocabulary')
		vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)
		save_vocab(vocab_index_dict, vocab_file, encoding)
		logging.info('Vocabulary is saved in %s', vocab_file)

	logging.info('Vocab size: %d', vocab_size)
	params['vocab_size'] = vocab_size

	# Create batch generators.
	train_batches = BatchGenerator(train_text, batch_size, num_unrollings, vocab_size,
								   vocab_index_dict, index_vocab_dict)
	valid_batches = BatchGenerator(valid_text, batch_size, num_unrollings, vocab_size,
								   vocab_index_dict, index_vocab_dict)
	test_batches = BatchGenerator(test_text, 1, 1, vocab_size,
								  vocab_index_dict, index_vocab_dict)

	# Create graphs
	logging.info('Creating graph')
	graph = tf.Graph()
	with graph.as_default():
		with tf.name_scope('training'):
			train_model = CharRNN(is_training=True, use_batch=True, **params)
		tf.get_variable_scope().reuse_variables()
		with tf.name_scope('validation'):
			valid_model = CharRNN(is_training=False, use_batch=True, **params)
		with tf.name_scope('evaluation'):
			test_model = CharRNN(is_training=False, use_batch=False, **params)
			saver = tf.train.Saver(name='checkpoint_saver', max_to_keep=max_to_keep)
			best_model_saver = tf.train.Saver(name='best_model_saver')

	logging.info('Model size (number of parameters): %s\n', train_model.model_size)
	logging.info('Start training\n')

	vocab_file = os.path.join(init_dir, 'vocab.json')

	# Initialize a timer
	stopwatch = 0

	try:
		# Use try and finally to make sure that intermediate
		# results are saved correctly so that training can
		# be continued later after interruption.
		with tf.Session(graph=graph) as session:
			graph_info = session.graph

			train_writer = tf.summary.FileWriter(tb_log_dir + 'train/', graph_info)
			valid_writer = tf.summary.FileWriter(tb_log_dir + 'valid/', graph_info)

			# load a saved model or start from random initialization.
			if init_model:
				saver.restore(session, os.path.join(save_path, init_model))
				logging.info('Restoring model: ' + init_model)
				session.run(tf.variables_initializer([tf.get_variable('global_step')]))
			else:
				tf.global_variables_initializer().run()
			for i in range(num_epochs):
				startTime = time.time()

				for j in range(n_save):
					logging.info('=' * 19 + ' Epoch %d: %d/%d' + '=' * 19 + '\n', i+1, j+1, n_save)
					logging.info('Training on training set')
					# training step
					epoch_data = train_model.run_epoch(
						session,
						train_size,
						train_batches,
						is_training=True,
						verbose=verbose,
						freq=progress_freq,
						divide_by_n=n_save)

					if epoch_data is not None:
						ppl, train_summary_str, global_step, learning_rate = epoch_data
					else:
						return

					# record the summary
					train_writer.add_summary(train_summary_str, global_step)
					train_writer.flush()
					# save model
					latest_model = saver.save(session, save_path + '/model', global_step=train_model.global_step)
					latest_model = latest_model[latest_model.rfind('/')+1:]
					logging.info('Latest model is %s\n', latest_model)
					logging.info('Evaluate on validation set')

					# valid_ppl, valid_summary_str, _ = valid_model.run_epoch(
					epoch_data = valid_model.run_epoch(
						session,
						valid_size,
						valid_batches,
						is_training=False,
						verbose=verbose,
						freq=progress_freq)

					if epoch_data is not None:
						valid_ppl, valid_summary_str, _, _ = epoch_data
					else:
						return

					# save and update best model
					if (not best_model) or (valid_ppl < best_valid_ppl):
						best_model = best_model_saver.save(
							session,
							best_path + '/model',
							global_step=train_model.global_step)
						best_model = best_model[best_model.rfind('/')+1:]
						best_valid_ppl = valid_ppl

					valid_writer.add_summary(valid_summary_str, global_step)
					valid_writer.flush()
					logging.info('Best model is %s', best_model)
					logging.info('Best validation ppl is %f\n', best_valid_ppl)
					result['latest_model'] = latest_model
					result['best_model'] = best_model

					# Convert to float because numpy.float is not json serializable.
					result['best_valid_ppl'] = float(best_valid_ppl)
					result['params']['max_learning_rate'] = float(learning_rate)

					result_path = os.path.join(output_dir, 'result.json')
					if os.path.exists(result_path):
						os.remove(result_path)
					with open(result_path, 'w') as f:
						json.dump(result, f, indent=2, sort_keys=True)

				stopTime = time.time()
				deltaTime = stopTime - startTime
				stopwatch += deltaTime
				logging.info('Time for last Epoch: %f', deltaTime)
				logging.info('Total elapsed time: %f\n', stopwatch)

			logging.info('Latest model is %s', latest_model)
			logging.info('Best model is %s', best_model)
			logging.info('Best validation ppl is %f\n', best_valid_ppl)
			logging.info('Evaluate the best model on test set')
			saver.restore(session, os.path.join(best_path, best_model))
			test_ppl, _, _, _ = test_model.run_epoch(session, test_size, test_batches,
												   is_training=False,
												   verbose=verbose,
												   freq=progress_freq)
			result['test_ppl'] = float(test_ppl)

	finally:
		result_path = os.path.join(output_dir, 'result.json')
		if os.path.exists(result_path):
			os.remove(result_path)
		with open(result_path, 'w') as f:
			json.dump(result, f, indent=2, sort_keys=True)

def create_vocab(text):
	unique_chars = list(set(text))
	vocab_size = len(unique_chars)
	vocab_index_dict = {}
	index_vocab_dict = {}
	for i, char in enumerate(unique_chars):
		vocab_index_dict[char] = i
		index_vocab_dict[i] = char
	return vocab_index_dict, index_vocab_dict, vocab_size


def load_vocab(vocab_file, encoding):
	with codecs.open(vocab_file, 'r', encoding=encoding) as f:
		vocab_index_dict = json.load(f)
	index_vocab_dict = {}
	vocab_size = 0
	for char, index in iteritems(vocab_index_dict):
		index_vocab_dict[index] = char
		vocab_size += 1
	return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file, encoding):
	with codecs.open(vocab_file, 'w', encoding=encoding) as f:
		json.dump(vocab_index_dict, f, indent=2, sort_keys=True)
