#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
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


# load mfcc from file
def _load_mfcc(mfcc_file):		
	# load mfcc
	mfcc = np.load(mfcc_file, allow_pickle=False)

	# speed perturbation augmenting
	mfcc = _augment_speech(mfcc)

	return mfcc


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

	def __init__(self, batch_size=16, n_mfcc=20, set_name='train'):

		# load meta file
		label, mfcc_file = [], []
		self.label_t, self.mfcc_t = [], []
		with open(data_path + 'preprocess_vn/meta/%s.csv' % set_name) as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:		# 11658 rows (=files)
				# mfcc file list
				filename = data_path + 'preprocess_vn/mfcc/' + row[0] + '.npy'
				mfcc_file.append(filename)

				# label info (convert to string object for variable-length support)
				info = np.asarray(row[1:], dtype=np.int)
				label.append(info.tostring())

		# decode string to integer
		for label_id in label:
			self.label_t.append(np.fromstring(label_id, np.int))
		
		# load mfcc from file
		for f_id in mfcc_file:
			mfcc = _load_mfcc(f_id)
			self.mfcc_t.append(mfcc)

		# create mini batches
#		self._create_batches(batch_size, n_mfcc)

		# reset pointer 
#		self._reset_batch_pointer()


	def _create_batches(self, batch_size=1, n_mfcc=20):
		# calculate total batch count
		self.n_batches = len(label) // batch_size	# Floor division

		# trim smaller final batch 
		self.mfcc_t  = self.mfcc_t[ :self.n_batches*self.batch_size]
		self.label_t = self.label_t[:self.n_batches*self.batch_size]

		# random shuffle data
		data = []
		for i in range(len(self.mfcc_t)):
			data.append([self.mfcc_t[i], self.label_t[i]])

		random.shuffle(data)
		
		# clear ordered data and assign shuffled one to tensors
		self.mfcc_t, self.label_t = [], []
		for i in range(data):
			self.mfcc_t.append(data[i][0])
			self.label_t.append(data[i][1])

		# calculate max length of mfcc & label
		self.mfcc_max_len  = max(len(mfcc)  for mfcc  in self.mfcc_t )
		self.label_max_len = max(len(label) for label in self.label_t)

		# create mini batch
		self.mfcc_batches, self.label_batches = [], []
		for i in range(self.n_batches):
			# Set start - stop indices
			start_index = i * self.batch_size
			stop_index = start_index + self.batch_size

			# trim batch
			mfcc_batch = self.mfcc_t[start_index:stop_index]
			label_batch = self.label_t[start_index:stop_index]

			# 0-padding
			for mfcc in mfcc_batch:
				while len(mfcc) < self.mfcc_max_len:
					mfcc.append([0]*self.n_mfcc)

			for label in label_batch:
				while len(label) < self.label_max_len:
					label.append(0)

			# add to batches
			self.mfcc_batches.append(mfcc_batch)
			self.label_batches.append(label_batch)

	
	def _next_batch(self):
		mfcc, label = self.mfcc_batches[self.pointer], self.label_batches[self.pointer]
		self.pointer += 1
		return mfcc, label


	def _reset_batch_pointer(self):
		self.pointer = 0



















