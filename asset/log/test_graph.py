#!/usr/bin/env python
#!/usr/bin/python2
# -*- coding: utf-8 -*-

# File: test_graph.py
# Date: Fri Sep 07 2018 GMT+07
# Author: Hari Yu <hari.wu.95@gmail.com>

### Import necessary libraries
import os, sys, glob, argparse, time
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
print("You are using tensorflow version " + tf.__version__)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("Sorry, tflearn is not ported to tensorflow 0.12 on windows yet!")
	quit()

from tensorflow.python.tools import inspect_checkpoint as ckpt 


### Define methods




### Test
print('\nTest nodes of wavenet graph')

# reset graph
tf.reset_default_graph()

# load checkpoint
with tf.Session() as sess:
	# load or restore checkpoint
	checkpoint_dir = './train_vn/tf'
	checkpoint = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)

	if (checkpoint):
		# Print tensors in checkpoint file
		print('Loading model from ' + checkpoint)
		num_blocks = 3
		for i in range(num_blocks):
			for r in [1, 2, 4, 8, 16]:
				block = 'block_%d_%d/' % (i,r)
				for n in ['conv_filter', 'conv_gate', 'conv_out']:
					ckpt.print_tensors_in_checkpoint_file(file_name=checkpoint, tensor_name=block+n+'/W', all_tensors=False)
					time.sleep(5)

		saver = tf.train.import_meta_graph(checkpoint+'.meta')
		saver.restore(sess, checkpoint)

		graph = tf.get_default_graph()
	
		# Get collection of global variables
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)		# TRAINABLE_VARIABLES
		print('\nThere are ', len(all_vars), ' vars in graph.')
		for var in all_vars:
			var_ = sess.run(var)
			print(var)
#			print(var_)

		# Print
		print('Finish loading!')

	else:
		print('Cannot load meta graph as well as SpeakersNet info.')
		time.sleep(3)
		exit()










