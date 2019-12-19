import sys
import os

sys.path.insert(0, "/home/recsys/guofangfang/tensor-demo/models-master/tutorials/rnn/ptb/")

import time
import numpy as np
import tensorflow as tf
import reader

class PTBInput(object):
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps ### RNN 的展开步骤
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

class PTBModel(object):
	def __init__(self, is_training, config, input_):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

	def lstm_cell():
		return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

	

	attn_cell = lstm_cell
	if is_training and config.keep_prob  < 1:
		def attn_cell():
			return tf.contrib.rnn.DropoutWrapper(
						lstm_cell(), output_keep_prob=config.keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell(
						[attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
	
	## 设置LSTM单元的 初始化状态为0.
	self._initial_state = cell.zero_state(batch_size, tf.float32)
	## LSTM单元，可以读入一个单词，并结合之前储存的状态state,计算下一个单词出现的概率分布,
	## 并且每次读取一个单词后它的状态state会被更新。
	
	## 词嵌入部分，即将one-hot的编码格式的单词转化为向量表达形式。word2vec中讲到。
	## 这部分，在GPU中还没有很好的实现，所以限定在CPU上运行。
	with tf.device("/cpu:0"): ##将计算限定在CPU中进行。
		## embedding matrix
		## vector size is hidden_size, same with hidden_size in LSTM cell
		embedding = tf.get_variable(
				"embedding", [vocab_size, size], dtype=tf.float32)
		inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
	
	if is_training and config.keep_prob < 1:
		inputs = tf.nn.dropout(inputs, config.keep_prob)
	
	
	outputs = []
	state = self._initial_state
	with tf.variable_scope("RNN"):
		for time_step in range(num_steps):
			if time_step > 0: 
				tf.get_variable_scope().reuse_varibale()
			(cell_output, state) = cell(inputs[:, time_step, :], state)
			outputs.append(cell_output)

	
	## 
	output = tf.reshape(tf.concat(outputs, 1), [-1, size])
	softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
	softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
	
	logits = tf.matmul(output, softmax_w) + softmax_b
	loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], 
						[tf.reshape(input_.targets, [-1])], 
						[tf.ones([batch_size * num_steps], dtype=tf.float32)])
	
	self._cost = cost = tf.reduce_sum(loss) / batch_size
	self._final_state = state

	if not is_training:
		return

	
	#### 学习速率
