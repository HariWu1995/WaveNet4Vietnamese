#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sugartensor as tf
import numpy as np
import re as regex	# REGular EXpression
import csv, string, codecs, random, librosa, time
from collections import Counter
from six.moves	 import cPickle, reduce, map

import sys
try:
	if sys.version_info[0] == '3':		# Python3
		sys.stdin  = sys.stdin.detach()
		sys.stdout = sys.stdout.detach()
	elif sys.version_info[0] == '2':	# Python2
		sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
		sys.stdin  = codecs.getreader('utf_8')(sys.stdin)
except: 
	print('Cannot handle utf-8 with sys.stdin/stdout')


__author__ = 'hari.wu.95@gmail.com'


# default data path
data_path = 'asset/data/'


# vocabulary table
index2byte = [	# 91 characters
				u' ',												# [0]
				u'A', u'À', u'Á', u'Ã', u'Ả', u'Ạ', 				# [1]
				u'Ă', u'Ằ', u'Ắ', u'Ẵ', u'Ẳ', u'Ặ',
				u'Â', u'Ầ', u'Ấ', u'Ẫ', u'Ẩ', u'Ậ',
				u'B', 												# [19]
				u'C', # u'CH',
				u'D', 												# [21] 
				u'Đ',
				u'E', u'È', u'É', u'Ẽ', u'Ẻ', u'Ẹ',					# [23]
				u'Ê', u'Ề', u'Ế', u'Ễ', u'Ể', u'Ệ',
				u'G', # u'GH', u'GI', 								# [35]
				u'H', 
				u'I', u'Ì', u'Í', u'Ĩ', u'Ỉ', u'Ị',	# u'IA', u'IÊ',
				u'K', # u'KH',										# [43]
				u'L', 
				u'M', 
				u'N', # u'NH', u'NG', u'NGH',						
				u'O', u'Ò', u'Ó', u'Õ', u'Ỏ', u'Ọ',					# [47]
				u'Ô', u'Ồ', u'Ố', u'Ỗ', u'Ổ', u'Ộ',
				u'Ơ', u'Ờ', u'Ớ', u'Ỡ', u'Ở', u'Ợ',
				u'P', # u'PH',										# [65]
				u'Q', # u'QU',
				u'R', 
				u'S', 
				u'T', # u'TH', u'TR',								# [69]
				u'U', u'Ù', u'Ú', u'Ũ', u'Ủ', u'Ụ', # u'UA', u'UÔ',
				u'Ư', u'Ừ', u'Ứ', u'Ữ', u'Ử', u'Ự', # u'ưa', u'ươ',
				u'V', 												# [82]
				u'X', 
				u'Y', u'Ỳ', u'Ý', u'Ỹ', u'Ỷ', u'Ỵ', # u'YÊ',
				'<EMP>'		# EoS - End of String					# [90]
]

for i, char in enumerate(index2byte):
	index2byte[i] = char.lower()


# accents table
accents = ['̀', '́', '̃', '̉', '̣']


# byte-to-index mapping
byte2index = {}
for i, char in enumerate(index2byte):
	byte2index[char] = i


# vocabulary size
vocab_size = len(index2byte)
# print('#vocab', vocab_size)


# convert sentence to index list
def str2index(str_):
	# clean white space
	str_ = u' '.join(str_.split())

	# remove punctuations like ',', '.', '?', '!', etc
#	str_ = str_.translate(None, string.punctuation)	# Python2
	str_ = str_.translate(str.maketrans(u'', u'', string.punctuation))	# Python3

	# make lower case
	str_ = str_.lower()

	res = []
	for char in str_:
#		print(char, len(char), type(char))
		try:
			idx = byte2index[char]
		except KeyError:
			# drop OOV (Out-Of-Vocabulary)
			pass
		res.append(idx)		
	return res


# Convert accent to non-accent Vietnamese
def de_accent_vnese(s, mode='1-byte-keystroke'):
	s = s.lower()

	s = regex.sub(u'[àáạảã]', 'a', s)
	s = regex.sub(u'[ầấậẩẫ]', 'â', s)
	s = regex.sub(u'[ằắặẳẵ]', 'ă', s)
	s = regex.sub(u'[èéẹẻẽ]', 'e', s)
	s = regex.sub(u'[ềếệểễ]', 'ê', s)
	s = regex.sub(u'[òóọỏõ]', 'o', s)
	s = regex.sub(u'[ồốộổỗ]', 'ô', s)
	s = regex.sub(u'[ờớợởỡ]', 'ơ', s)
	s = regex.sub(u'[ìíịỉĩ]', 'i', s)
	s = regex.sub(u'[ùúụủũ]', 'u', s)
	s = regex.sub(u'[ừứựửữ]', 'ư', s)
	s = regex.sub(u'[ỳýỵỷỹ]', 'y', s)

	if mode == 'ascii':
		s = regex.sub(u'đ' , 'd', s)
	return s


# convert index list to string
def index2str(index_list):
	# transform label index to character
	str_ = u''
	for char in index_list:
		if char < (vocab_size-1):
			str_ += index2byte[char]
		elif char == (vocab_size-1):  # <EOS>
			break
	return str_


# print list of index list
def print_index(indices):
	for index_list in indices:
		print(index2str(index_list))


# real-time wave to mfcc conversion function
@tf.sg_producer_func
def _load_mfcc(src_list):		

	# label, wave_file
	label, mfcc_file = src_list

	# decode string to integer
	label = np.fromstring(label, np.int)

	# load mfcc
	mfcc = np.load(mfcc_file, allow_pickle=False)

	# speed perturbation augmenting
	mfcc = _augment_speech(mfcc)

	return label, mfcc


def _augment_speech(mfcc):
	# random frequency shift ( == speed perturbation effect on MFCC )
	r = np.random.randint(-2, 2)

	# shifting mfcc
	mfcc = np.roll(mfcc, r, axis=0)

	# zero padding
	if r > 0:
		mfcc[:r, :] = 0
	elif r < 0:
		mfcc[r:, :] = 0

	return mfcc


# Speech Corpus
class SpeechCorpus(object):

	def __init__(self, batch_size=16, set_name='train'):

		# load meta file
		label, mfcc_file = [], []
		with open(data_path + 'preprocess_vn/meta/%s.csv' % set_name) as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:		# 11658 rows (=files)
				# mfcc file list
				filename = data_path + 'preprocess_vn/mfcc/' + row[0] + '.npy'
				mfcc_file.append(filename)

				# label info (convert to string object for variable-length support)
				info = np.asarray(row[1:], dtype=np.int)
				label.append(info.tostring())

		# to constant tensor
		label_t     = tf.convert_to_tensor(label)
		mfcc_file_t = tf.convert_to_tensor(mfcc_file)

		# create queue from constant tensor
		label_q, mfcc_file_q = tf.train.slice_input_producer(tensor_list=[label_t, mfcc_file_t], shuffle=True, capacity=32)

		# create label, mfcc queue
		label_q, mfcc_q = _load_mfcc(source=[label_q, mfcc_file_q],
										dtypes=[tf.sg_intx, tf.sg_floatx],
										capacity=256, 
										num_threads=64)

		# create batch queue with dynamic padding
		batch_queue = tf.train.batch([label_q, mfcc_q], 
										batch_size,
										shapes=[(None,), (20, None)],
										num_threads=64, 
										capacity=batch_size*32,
										dynamic_pad=True,
										allow_smaller_final_batch=True)

		# split data
		self.label, self.mfcc = batch_queue

		# batch * time * dim
		self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])

		# calculate total batch count
		self.num_batch = len(label) // batch_size	# Floor division

		# print info
		tf.sg_info('%s set loaded.(total data=%d, total batch=%d)' % (set_name.upper(), len(label), self.num_batch))





