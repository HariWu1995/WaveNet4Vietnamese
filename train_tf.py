#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	# TF environment variable responsible for logs
											# Set to 1 in 3 values below:
											# 		1: silence INFO logs
											# 		2: filter out WARNING logs
											# 		3: silence ERROR logs (not recommended)
import time
import tensorflow as tf		# require version 1.0.0
from tensorflow.python.client import device_lib

from data_vn import SpeechCorpus, vocab_size
from model_tf import WaveNet


__author__ = 'hari.wu.95@gmail.com'


def _init_device():
	# set log level to debug
	tf.logging.set_verbosity(10)

	# get total #GPUs available
	local_device_protos = device_lib.list_local_devices()
	_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
	return max(_gpus, 1)


def _load_data(batch_size, gpu):
	print("Loading speech corpus")
	# load corpus input tensor
	data = SpeechCorpus(batch_size=batch_size*gpu)

	# split inputs for each GPU tower
	inputs = tf.split(data.mfcc, gpu, axis=0, name='inputs')	# mfcc features of audio
	labels = tf.split(data.label, gpu, axis=0, name='labels')	# target sentence label

	# sequence length except zero-padding
	seq_len = []
	for input_ in inputs:
		seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))

	print("	inputs\n	", inputs)
	print("	labels\n	", labels)
	return inputs, labels, seq_len, data.num_batch


def train(batch_size=16, max_epoch=100000, n_mfcc=20, n_char=vocab_size):
	# initialise requirements
	gpu = _init_device()
	print("Finish initialising requirements")
	
	# load model
	model = WaveNet(n_out=vocab_size, batch_size=batch_size, n_features=n_mfcc, onTrain=True)
	print("Finish loading WaveNet")

	# train
#	config = tf.ConfigProto(inter_op_parallelism_threads=1,
#							intra_op_parallelism_threads=1,
#							allow_soft_placement=True)
#	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	with tf.Session() as sess:
		# initialise variables, constants
		sess.run([	tf.local_variables_initializer(),
					tf.global_variables_initializer()	#, tf.constant_initializer()
				])
		print("Finish initialising variables")
		
		# restore model (if any)
		checkpoint_dir = 'asset/log/train_vn/tf'
		checkpoint = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)
		if (checkpoint):
			print('Load from last model')
			saver = tf.train.import_meta_graph(checkpoint+'.meta')
			saver.restore(sess, checkpoint)
		else:
			print('Cannot find any model')
			saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=7, keep_checkpoint_every_n_hours=1, name='Saver', write_version=tf.train.SaverDef.V2)

		# save the graph for Tensorboard visualization
		graph = tf.get_default_graph()
		tf.train.write_graph(sess.graph_def, logdir='./asset/log/train_vn/tf/', name='wavenet-model.pbtxt')

		# training
		print("\nModel on train ...")
		train_loss = 100
		for epoch in range(max_epoch):
			print("epoch %d" % epoch)
			# load data
			inputs, labels, seq_len, num_batch = _load_data(batch_size, gpu)

			for batch in range(num_batch):
				print("batch %d" % batch)
				# convert tensor to np array
				model.inputs  = inputs
				model.targets = labels
				model.seq_len = seq_len

				# run optimization (backprop) & calculate batch loss
				_, train_loss = sess.run([model.optimizer_op, model.cost])

				# display
				print("epoch: %d/%d, batch: %d/%d, batch_loss: %s." % (epoch, max_epoch, batch, num_batch, train_loss))

			# check for conditional stop
			if train_loss < 1e-7:
				break

		# save a checkpoint file of the model variables
		saver.save(sess, './asset/log/wavenet-model.ckpt', global_step=epoch)


if __name__ == "__main__":
	print(__author__)
	train()
	print("Finish training")

















