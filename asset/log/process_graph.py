#!/usr/bin/python2
# -*- coding: utf-8 -*-

# File: graph_freezer.py
# Date: Mon Jul 30 2018 GMT+07
# Author: Hari Yu <hari.wu.95@gmail.com>

### Import necessary libraries
import os, argparse, time, glob

import tensorflow as tf
from tensorflow.core.framework import graph_pb2, node_def_pb2
from tensorflow.python.tools   import freeze_graph 
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference 		  as optimize4infer
from tensorflow.python.framework.graph_util 			import convert_variables_to_constants as vars2consts

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


### Declare arguments and freeze the graph
dir_name = os.path.dirname(os.path.realpath(__file__))
model_name = 'wavenet-model'


### Define method 
def graph_freezer(model_dir, output_node_names):
	""" Extract the sub graph defined by the output nodes and convert all its vars into consts.
	It's useful to when we need to load a single file in C++, especially in environments like mobile 
	or embedded where we may not have access to the RestoreTensor ops and file loading calls.
	Args:
		model_dir: the root folder containing the checkpoint state file
		output_node_names: a string, containing all the output node's names separated by comma
	"""

	# Check input arguments
	if not tf.gfile.Exists(model_dir):
		# raise AssertionError("Export directory doesn't exist. Please specify an export directory: %s" % model_dir)
		os.makedirs('archive')
		model_dir = 'archive'

	if not output_node_names:
		print("You need to supply the name of a node to --output_node_names.")
		return -1

	# Retrieve checkpoint full-path
	checkpoint = tf.train.get_checkpoint_state(model_dir)
	input_checkpoint = checkpoint.model_checkpoint_path
	print('Extract data from checkpoint: ', input_checkpoint)

	# Precise frozen graph file fullname
	abs_model_dir = "/".join(input_checkpoint.split('/')[:-1])
	frozen_graph_name = abs_model_dir + "/frozen_" + model_name + ".pb"

	# clear devices to allow TF control on which device it will load operations
	clear_devices = True
	
	# start a session using a temporary fresh Graph
	with tf.Session(graph=tf.Graph()) as sess:
		# import the meta graph in the current default Graph
		saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices)

		# restore the weights & biases
		saver.restore(sess, input_checkpoint)

		# use a built-in TF utility to export variables to constants
		graph_def = tf.get_default_graph().as_graph_def()
		display_nodes(graph_def.node)
		frozen_graph_def = vars2consts(	sess, 		# retrieve the weights
										graph_def, 	# retrieve the nodes
										output_node_names.replace(" ", "").split(",") )	# select the usefull nodes

		# serialize and dump the output graph to the filesystem
		with tf.gfile.GFile(frozen_graph_name, "wb") as gf:
			gf.write(frozen_graph_def.SerializeToString())
		
	print('Finish freezing graph!')
	print("%d ops in the final graph." % len(frozen_graph_def.node))
	return frozen_graph_def, frozen_graph_name


def graph_loader(frozen_graph_filename):
	# load protobuf file from disk & parse it to retrieve the unserialized graph_def
	with tf.gfile.GFile(frozen_graph_filename, "rb") as gf:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(gf.read())

	# import the graph_def into a new Graph
	with tf.Graph().as_default() as graph:
		# The name var will prefix every op/nodes in your graph
		# Since we load everything in a new graph, this is not needed
		tf.import_graph_def(graph_def, name="prefix")

	return graph


def graph_optimizer(frozen_graph_name, inNodes, outNodes):
	""" Removes parts of a graph that are only needed for training.
	There are several common transformations that can be applied to GraphDefs created to train a model, 
	that help reduce the amount of computation needed when network is used only for inference, including:
		 - Remove training-only ops like checkpoint saver.
		 - Strip out parts of graph that are never reached.
		 - Remove debug ops like CheckNumerics.
		 - Fold batch normalization ops into the pre-calculated weights.
		 - Fuse common ops into unified versions.
	This method takes a frozen GraphDef file (where the weight vars converted into consts by graph_freezer) 
	and outputs a new GraphDef with the optimizations applied.
	"""
	# read data from frozen graph
	optimized_graph_name   = frozen_graph_name.replace('frozen', 'optimized')
	dropoutless_graph_name = frozen_graph_name.replace('frozen', 'dropoutless')
	print(frozen_graph_name, ' --> ', optimized_graph_name, ' --> ', dropoutless_graph_name)
	with tf.gfile.GFile(frozen_graph_name, "rb") as gf:
		input_graph_def = tf.GraphDef()
		input_graph_def.ParseFromString(gf.read())

	# optimize w/o removing dropout
	optimized_graph_def = optimize4infer(input_graph_def, input_node_names=inNodes, output_node_names=outNodes, placeholder_type_enum=tf.float32.as_datatype_enum)

	# remove dropout
	dropoutless_graph_def = dropout_remover(optimized_graph_def)

	# save optimized & dropoutless graphs
	gf = tf.gfile.FastGFile(optimized_graph_name, "wb")
	gf.write(optimized_graph_def.SerializeToString())

	gf_ = tf.gfile.FastGFile(dropoutless_graph_name, "wb")
	gf_.write(dropoutless_graph_def.SerializeToString())

	print('Finish optimizing graph!')

	return optimized_graph_def, optimized_graph_name, dropoutless_graph_def, dropoutless_graph_name


def display_nodes(nodes):
	for idx, node in enumerate(nodes):				# main nodes
		print('%d %s %s' % (idx, node.name, node.op))
		for _idx, _node in enumerate(node.input):	# sub-nodes
			print(u'└─── %d ─ %s' % (_idx, _node))


def dropout_remover(input_graph):
	"""	Remove dropout layer & connect nodes before dropout with nodes after dropout.
	"""
	nodes_bfore = input_graph.node
	nodes_after = []
	nodes_bfore_dropout = []
	nodes_after_dropout = []
	
	# Initialise
	for node in nodes_bfore:
#		print('Node: ', node.name)
		if node.name.startswith('Dropout/'):	# Find nodes before dropout
#			print('sub-nodes: ', node.input)
			for n in node.input:
				if not (n.startswith('Dropout/') or n.startswith('keep_prob/')):
#					print('	Add node', n, 'to nodes_bfore_dropout')
					nodes_bfore_dropout.append(n)
		else:	# Find nodes after dropout
#			print('sub-nodes: ', node.input)
			for n in node.input:
				if n.startswith('Dropout/'):
#					print('	Add node', n, 'to nodes_after_dropout')
					nodes_after_dropout.append(node.name)
	
	# remove already-existing nodes
	nodes_bfore_dropout = list(set(nodes_bfore_dropout))
	nodes_after_dropout = list(set(nodes_after_dropout))

	print('Nodes bfore dropout: ', nodes_bfore_dropout)
	print('Nodes after dropout: ', nodes_after_dropout)

	# Processing
	idx = 0
	for node in nodes_bfore:
		# skip dropout nodes
		if node.name.startswith('Dropout/') or node.name.startswith('keep_prob/'):
			continue

		new_node = node_def_pb2.NodeDef()
		new_node.CopyFrom(node)
		
		if any (new_node.name in n for n in nodes_after_dropout):
			new_input = []
			for sub_node in new_node.input:
				if sub_node.startswith('Dropout/'):
					new_input.append(nodes_bfore_dropout[0])
				else:
					new_input.append(sub_node)
			del new_node.input[:]
			new_node.input.extend(new_input)

		nodes_after.append(new_node)
	
	output_graph = graph_pb2.GraphDef()
	output_graph.node.extend(nodes_after)

	print('Finish removing dropout!')
	return output_graph


### Main
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir", type=str, default="./train_vn_backup", help="Model folder to export", required=False)

	default_nodes = "front/conv_in/W, logit/conv_1/W, logit/conv_2/W, logit/conv_2/b, "
	num_blocks = 3
	for i in range(num_blocks):
		for r in [1, 2, 4, 8, 16]:
			block = 'block_%d_%d/' % (i,r)
			for n in['conv_filter', 'conv_gate', 'conv_out']:
				default_nodes += block + n + '/W' + ", "
	default_nodes = default_nodes[0:len(default_nodes)-2]
	for n in default_nodes.replace(" ", "").split(','):
		print(n)
	parser.add_argument("--output_node_names", type=str, default=default_nodes, help="Names of the output nodes, separated by ','.", required=False)
	
	args = parser.parse_args()
	
	# freeze graph
	frozen_graph_def, frozen_graph_name = graph_freezer(args.model_dir, args.output_node_names)

	# load frozen graph
	graph = graph_loader(frozen_graph_name)

	ops = []
	for op in graph.get_operations():
		print(op.name)
		ops.append(op.name)

	# access the nodes
#	ind_x = ops.index('prefix/front/conv_in/W')
#	x = graph.get_tensor_by_name(ops[ind_x] + ':0')
	

	# optimize graph
	optimized_graph_def, optimized_graph_name, dropoutless_graph_def, dropoutless_graph_name = graph_optimizer(frozen_graph_name, inNodes=['front/conv_in/W'], outNodes=['logit/conv_1/W', 'logit/conv_2/W'])

	# test graph
	graph = tf.GraphDef()
	filename = 'train_vn/' + 'frozen' + '_wavenet-model.pb'
	with tf.gfile.Open(filename, 'rb') as gf:
		graph.ParseFromString(gf.read())
	print('Display nodes in ' + filename)
	display_nodes(graph.node)



