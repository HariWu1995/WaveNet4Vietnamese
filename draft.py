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

from data_vn_tf import SpeechCorpus, vocab_size
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


def train(batch_size=16, max_epoch=100, n_mfcc=20, n_char=vocab_size):
	# initialise requirements
	gpu = _init_device()
	print("Finish initialising requirements")
	
	# load model
	model = WaveNet(n_out=vocab_size, batch_size=batch_size, n_features=n_mfcc, onTrain=True)
	print("Finish loading WaveNet")

	# load data & create dataset
	data = SpeechCorpus(batch_size=16)
	features, labels = data.mfcc_t, data.label_t
	dataset = tf.data.Dataset.from_tensor_slices((features,labels))
	dataset.shuffle(buffer_size=100).batch(batch_size)
	iter = dataset.make_one_shot_iterator()
	model.inputs, model.targets = iter.get_next()

	# train
	with tf.Session() as sess:
		# initialise variables, constants
		sess.run([	tf.local_variables_initializer(),
					tf.global_variables_initializer()	#, tf.constant_initializer()
				])
		print("Finish initialising variables")

		# training
		print("\nModel on train ...")
		for epoch in range(max_epoch):
			for batch in range(num_batch):
				# run optimization (backprop) & calculate batch loss
				_, train_loss = sess.run([model.optimizer_op, model.cost])

				# display
				print("epoch: %d/%d, batch: %d/%d, batch_loss: %s." % (epoch, max_epoch, batch, num_batch, train_loss))


if __name__ == "__main__":
	print(__author__)
	train()
	print("Finish training")

















