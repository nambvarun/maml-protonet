import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ProtoNet(tf.keras.Model):
	def __init__(self, num_filters, latent_dim):
		super(ProtoNet, self).__init__()
		self.num_filters = num_filters
		self.latent_dim = latent_dim
		num_filter_list = self.num_filters + [latent_dim]
		self.convs = []
		for i, num_filter in enumerate(num_filter_list):
			block_parts = [
				layers.Conv2D(
					filters=num_filter,
					kernel_size=3,
					padding='SAME',
					activation='linear'),
			]

			block_parts += [layers.BatchNormalization()]
			block_parts += [layers.Activation('relu')]
			block_parts += [layers.MaxPool2D()]
			block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
			self.__setattr__("conv%d" % i, block)
			self.convs.append(block)
		self.flatten = tf.keras.layers.Flatten()

	def call(self, inp):
		out = inp
		for conv in self.convs:
			out = conv(out)
		out = self.flatten(out)
		return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
	"""
		calculates the prototype network loss using the latent representation of x
		and the latent representation of the query set
		Args:
			x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
			q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
			labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
			num_classes: number of classes (N) for classification
			num_support: number of examples (S) in the support set
			num_queries: number of examples (Q) in the query set
		Returns:
			ce_loss: the cross entropy loss between the predicted labels and true labels
			acc: the accuracy of classification on the queries
	"""
	#############################
	#### YOUR CODE GOES HERE ####

	# compute the prototypes
	row_idx_to_sum = tf.range(num_classes)
	row_idx_to_sum = tf.reshape(row_idx_to_sum, [-1, 1])
	row_idx_to_sum = tf.tile(row_idx_to_sum, [1, num_support])
	row_idx_to_sum = tf.reshape(row_idx_to_sum, [-1])
	# row_idx_to_sum = [idx for idx in row_idx_to_sum for i in tf.range(num_support)]
	prototypes = tf.math.segment_sum(x_latent, row_idx_to_sum)
	prototypes /= tf.to_float(num_classes)												# paper says to divide by all classes
	# shape of prototypes is [N, D]

	# compute the distance from the prototypes
	labels_onehot = tf.reshape(labels_onehot, [-1, tf.shape(labels_onehot)[-1]])		# reshaping to [N*Q, N]
	# distances = tf.zeros_like(labels_onehot)		# shape of [N*Q, N]

	# let's try vectorizing this nonsense
	q_latent_repeated = tf.keras.backend.repeat(q_latent, num_classes)
	q_latent_repeated = tf.reshape(q_latent_repeated, [-1, tf.shape(q_latent)[-1]])

	prototypes_tiled = tf.tile(prototypes, (num_queries * num_classes, 1))

	distances = tf.linalg.norm(q_latent_repeated - prototypes_tiled, axis=-1)			# [N*N*Q, D] - [N*N*Q, D]
	distances = tf.reshape(distances, tf.shape(labels_onehot))

	# for i in range equivalent in tf
	# i = tf.constant(0)
	# while_condition = lambda i: tf.less(i, tf.shape(distances)[0])
	#
	# def add_distances():
	# 	distances[i] = tf.linalg.norm(q_latent[i] - prototypes, axis=-1)
	# 	return [tf.add(i, 1)]
	#
	# tf.while_loop(while_condition, add_distances, [i])

	# for i in tf.range(tf.shape(distances)[0]):
	# 	distances[i] = tf.linalg.norm(q_latent[i] - prototypes, axis=-1) 	# hoping the broadcasting will work.

	# compute cross entropy loss
	loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=-distances)
	ce_loss = tf.reduce_mean(loss)

	# note - additional steps are needed!
	acc = tf.math.equal(tf.argmax(tf.nn.softmax(logits=-distances), 1), tf.argmax(labels_onehot, 1))
	acc = tf.reduce_mean(tf.cast(acc, 'float'))

	# return the cross-entropy loss and accuracy
	# ce_loss, acc = None, None
	#############################
	return ce_loss, acc

