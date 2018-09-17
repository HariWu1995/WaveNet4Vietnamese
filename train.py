#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	# TF environment variable responsible for logs
											# Set to 1 in 3 values below:
											# 		1: silence INFO logs
											# 		2: filter out WARNING logs
											# 		3: silence ERROR logs (not recommended)
import os, argparse, time, glob
import sugartensor as tf

from model import *


__author__ = 'hari.wu.95@gmail.com'


def train(lang):
	# set log level to debug
	tf.sg_verbosity(10)

	# hyper-parameters
	gpu = tf.sg_gpus()		# batch size adjusted for multiple GPUs
	batch_size = 16    		# total batch size
	lr    = 0.0001
	beta1 = 0.9
	beta2 = 0.999
	max_epoch = 100
	n_features = 20

	# corpus input tensor
	data = SpeechCorpus(batch_size=batch_size*gpu)

	# split inputs for each GPU tower
	try:
		inputs = tf.split(value=data.mfcc, num_or_size_splits=gpu, axis=0)	# mfcc features of audio
		labels = tf.split(value=data.label, num_or_size_splits=gpu, axis=0)	# target sentence label
	except:
		inputs = data.mfcc
		labels = data.label

	# sequence length except zero-padding
	seq_len = []
	for input_ in inputs:
		seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))

	# train
	log_dir = 'asset/log/train_vn/sg' if lang == "vn" else 'asset/log/train'
	tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len), ep_size=data.num_batch, max_ep=500, save_dir=log_dir)


# parallel-training loss tower
@tf.sg_parallel		# {@} [decorator] software design pattern: dynamically alter functionality
					# of a FUNCTION, METHOD, or CLASS w/o having to directly use subclasses
					# or change source code of the function being decorated.
def get_loss(opt):
	# encode audio feature
	logit = get_logit(x=opt.input[opt.gpu_index], voca_size=vocab_size)

	# CTC loss
	return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index], name='ctc_loss')


if __name__ == "__main__":
	print(__author__)

	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", type=str, default="vn", help="Language to process", required=False)

	args = parser.parse_args()
	args.lang.lower()
	if args.lang == ("vn" or "vnese" or "vietnamese"):
		args.lang = "vn"
		from data_vn import SpeechCorpus, vocab_size
	elif args.lang == ("en" or "eng" or "english"):
		args.lang = "en"
		from data import SpeechCorpus, voca_size
		vocab_size = voca_size

	print("Load " + args.lang.upper() + " data corpus. \nStart training ")
	train(args.lang)











