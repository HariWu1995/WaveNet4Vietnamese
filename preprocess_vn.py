#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob, csv, librosa, os, subprocess, time
import numpy as np
import pandas as pd

import data_vn

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

__author__ = 'hari.wu.95@gmail.com'


# data path
data_path = "asset/data/"


#
# process Vivos corpus
#
def process_vivos(csv_file, category):
	parent_path = data_path + 'vivos/'
	labels, wave_files = [], []

	# create csv writer
	writer = csv.writer(csv_file, delimiter=',')

	# read label-info
	content_filename = parent_path + category + '/prompts.txt'
	label_info = pd.read_table(content_filename, usecols=['ID'], index_col=False, delim_whitespace=True)
	# print(label_info)		# testpoint: label_info

	# read file IDs
#	file_ids = []
#	for uid in label_info.ID.values:
#	print(uid)		# testpoint: uid
#	folder_path, filename = uid.split("_")
#	for d in [parent_path + category + '/waves/%s' % folder_path]:
#		print(d)		# testpoint: folder_path
#		a = glob.glob(d + '*.txt')
#		print(a)
#		b = sorted(glob.glob(d + '*.txt'))
#		print(b)
#		for f in sorted(glob.glob(d + '*.txt')):
#		# print(f[-12:-4])
#		file_ids.extend([f[-12:-4]])
#		# print(file_ids)
	file_ids = label_info.ID
	# print(file_ids)		# testpoint: file_ID


	# preprocess
	content_ = open(content_filename, 'r')
	title_content = content_.readline()		
#	print(title_content)		# Result: 'ID\t\tContent\n'
	for i, f in enumerate(file_ids):
		# wave file name
		wave_file = parent_path + category + '/waves/%s/' % f[0:10] + f + '.wav'
#		print(wave_file)        
		fn = wave_file.split('/')[-1]
#		print(fn)
		target_filename = 'asset/data/preprocess_vn/mfcc/' + fn + '.npy'
#		print(target_filename)
		if os.path.exists(target_filename):
			continue
		print("Vivos corpus preprocessing (%d/%d) - ['%s']" % (i, len(file_ids), wave_file))

		# load wave file
		wave, sr = librosa.load(wave_file, sr=16000, mono=True)	# default: sr=22050Hz

		# re-sample (48K --> 16K)
		# wave = wave[::3]

		# get mfcc feature
		mfcc = librosa.feature.mfcc(wave, sr=16000)

		# get label index
		curr_content = content_.readline()
		curr_content = curr_content[(len(fn)-3):(len(curr_content))]
		print(curr_content)
		label = data_vn.str2index(curr_content)

		# save result (exclude small mfcc data to prevent CTC loss)
		if len(label) < mfcc.shape[1]:
			# save meta info
			writer.writerow([fn] + label)

			# save mfcc
			np.save(target_filename, mfcc, allow_pickle=False)

			# check saved features
			print(data_vn.index2str(label), '\n')
		
		# delay for observation and analysis
		# time.sleep(10)

#
# Create directories
#
if not os.path.exists('asset/data/preprocess_vn'):
	os.makedirs('asset/data/preprocess_vn')
if not os.path.exists('asset/data/preprocess_vn/meta'):
	os.makedirs('asset/data/preprocess_vn/meta')
if not os.path.exists('asset/data/preprocess_vn/mfcc'):
	os.makedirs('asset/data/preprocess_vn/mfcc')

#
# Run pre-processing for training
#

# Vivos corpus for training
csv_file_train = open('asset/data/preprocess_vn/meta/train.csv', 'w')
process_vivos(csv_file_train, 'train')
csv_file_train.close()


#
# Run pre-processing for testing
#

# Vivos corpus for test
csv_file_test = open('asset/data/preprocess_vn/meta/test.csv', 'w')
process_vivos(csv_file_test, 'test')
csv_file_test.close()
