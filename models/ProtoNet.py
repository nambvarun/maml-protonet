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
	q_latent_repeated = tf.keras.backend.repeat(q_latent, num_classes)
	q_latent_repeated = tf.reshape(q_latent_repeated, [-1, tf.shape(q_latent)[-1]])

	prototypes = tf.reshape(x_latent, [num_classes, num_support, -1])
	prototypes = tf.reduce_mean(prototypes, axis=1)

	# compute the distance from the prototypes
	labels_onehot = tf.reshape(labels_onehot, [-1, tf.shape(labels_onehot)[-1]])
	prototypes_tiled = tf.tile(prototypes, (num_queries * num_classes, 1))
	distances = tf.square(tf.linalg.norm(q_latent_repeated - prototypes_tiled, axis=-1))
	distances = tf.reshape(distances, tf.shape(labels_onehot))

	# compute cross entropy loss

	# distances = tf.reshape(dists, [num_classes * num_queries, num_classes])
	labels_onehot_reshaped = tf.reshape(labels_onehot, [num_classes * num_queries, num_classes])
	ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot_reshaped, logits=-distances))

	log_p_y = tf.nn.log_softmax(-distances, axis=-1)
	# log_p_y = tf.reshape(log_p_y, [num_classes, num_queries, -1])
	# ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(labels_onehot, log_p_y), axis=-1), [-1]))

	# note - additional steps are needed!
	# eq = tf.cast(tf.equal(
	# 	tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
	# 	tf.cast(y, tf.int32)), tf.float32)
	# acc = tf.reduce_mean(eq)
	acc = tf.equal(tf.argmax(log_p_y, axis=-1), tf.argmax(labels_onehot, axis=-1))
	acc = tf.reduce_mean(tf.to_float(acc))

	# return the cross-entropy loss and accuracy
	# ce_loss, acc = None, None
	#############################
	return ce_loss, acc

