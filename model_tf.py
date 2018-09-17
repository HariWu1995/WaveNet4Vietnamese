#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

__author__ = 'hari.wu.95@gmail.com'


class WaveNet():
	
	def __init__(self, n_out, batch_size=16, n_features=20, onTrain=True):
		
		# hyper-parameters
		n_blocks = 3	# dilated blocks
		n_dim = 128		# latent dimension
		self.onTrain = onTrain

		# initialise in/out-puts
		self.inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, n_features], name="mfcc")
		self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, None], name="character")
		self.seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.inputs, reduction_indices=2), 0.), tf.int32), reduction_indices=1, name="seqLen")

		# 1D convolution
		self.conv1d_index = 0
		z = self.conv1d_layer(self.inputs, dim=n_dim)

		# stack hole atrous/dilated convolution
		skip = 0	# skip connections
		self.aconv1d_index = 0
		for block in range(n_blocks):
			for rate in [1, 2, 4, 8, 16]:
				z, s = self.residual_block(z, size=7, rate=rate, block=block, dim=n_dim)
				skip += s
		
		# final logit layers
		with tf.name_scope("logit"):
			with tf.name_scope("conv_1"):
				logit = self.conv1d_layer(skip, dim=skip.get_shape().as_list()[-1], activation='tanh')
			with tf.name_scope("conv_2"):
				self.logit = self.conv1d_layer(logit, dim=n_out, bias=True, activation=None)

		# CTC loss
		indices = tf.where(tf.not_equal(tf.cast(self.targets, tf.float32), 0.))
		target = tf.SparseTensor(indices=indices, values=tf.gather_nd(self.targets, indices)-1, dense_shape=tf.cast(tf.shape(self.targets), tf.int64))
		with tf.name_scope('Loss'):
			loss = tf.nn.ctc_loss(labels=target, inputs=self.logit, sequence_length=self.seq_len, time_major=False)
			loss = tf.reduce_mean(loss)
		self.cost = loss

		# Optimizer
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer()
			var_list = [var for var in tf.trainable_variables()]
			gradient = optimizer.compute_gradients(self.cost, var_list=var_list)
			optimizer = optimizer.apply_gradients(gradient)
		self.optimizer_op = optimizer


	def residual_block(self, tensor_in, size, rate, block, dim):
		layer_name = "block_%d_%d" % (block, rate)
		with tf.name_scope(layer_name):
			# filter convolution
			with tf.name_scope("conv_filter"):
				conv_filter = self.aconv1d_layer(tensor_in, size=size, rate=rate, activation='tanh')

			# gate convolution
			with tf.name_scope("conv_gate"):
				conv_gate = self.aconv1d_layer(tensor_in, size=size, rate=rate, activation='sigmoid')
	
			# output by gates multiplied
			out = conv_filter * conv_gate
			
			# final output
			with tf.name_scope("conv_out"):
				out = self.conv1d_layer(out, size=1, dim=dim, activation="tanh")

			# residual and skip output
			return out + tensor_in, out


	def conv1d_layer(self, tensor_in, size=1, dim=128, bias=False, activation='tanh'):
		layer_name = 'conv1d_' + str(self.conv1d_index)
		with tf.variable_scope(layer_name):
			shape = tensor_in.get_shape().as_list()
			kernel = tf.get_variable('kernel', (size, shape[-1], dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			if bias:
				b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
			out = tf.nn.conv1d(tensor_in, kernel, stride=1, padding='SAME') + (b if bias else 0)
			if not bias:
				out = self.batch_norm_wrapper(out)

			out = self.activation_wrapper(out, activation)
			
			self.conv1d_index += 1
			return out


	def aconv1d_layer(self, tensor_in, size=7, rate=2, bias=False, activation='tanh'):
		layer_name = 'aconv1d_' + str(self.aconv1d_index)
		with tf.variable_scope(layer_name):
			shape = tensor_in.get_shape().as_list()
			kernel = tf.get_variable('kernel',(1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			if bias:
				b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
			out = tf.nn.atrous_conv2d(tf.expand_dims(tensor_in, dim=1), kernel, rate=rate, padding='SAME')
			out = tf.squeeze(out, [1])
			if not bias:
				out = self.batch_norm_wrapper(out)

			out = self.activation_wrapper(out, activation)
			
			self.aconv1d_index += 1
			return out


	def batch_norm_wrapper(self, inputs, decay=0.999):
		epsilon = 1e-3
		shape = inputs.get_shape().as_list()

		beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
		gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
		pop_mean = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
		pop_var = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
		if self.onTrain:
			batch_mean, batch_var = tf.nn.moments(inputs, axes=list(range(len(shape)-1)))
			train_mean = tf.assign(pop_mean, pop_mean*decay+batch_mean*(1-decay))
			train_var  = tf.assign(pop_var ,  pop_var*decay+batch_var*(1-decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
		else:
			return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)


	def activation_wrapper(self, inputs, activation=None):
		out = inputs	# for null activation

		if activation == 'sigmoid':
			out = tf.nn.sigmoid(out)
		elif activation == 'tanh':
			out = tf.nn.tanh(out)
		elif activation == 'relu':
			out = tf.nn.relu(out)

		return out


























