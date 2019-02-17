"""Author: Brandon Trabucco, Copyright 2019
Implements an attention mechanism for images to replace convolutions.
MIT license.
"""


import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.logging.set_verbosity(tf.logging.INFO)
tf.flags.DEFINE_integer("batch_size", 32, "")
tf.flags.DEFINE_integer("num_cores", 8, "")
tf.flags.DEFINE_integer("num_epochs", 10, "")
FLAGS = tf.flags.FLAGS


TRAIN_DATASET_PATH = "D:/datasets/cifar-100-python/train"
TEST_DATASET_PATH = "D:/datasets/cifar-100-python/test"


def load_train_dataset_numpy():
    with open(TRAIN_DATASET_PATH, 'rb') as fo:
        result = pkl.load(fo, encoding='bytes')
    return {"data": result[b"data"], "labels": np.array(result[b"fine_labels"])}


def load_test_dataset_numpy():
    with open(TEST_DATASET_PATH, 'rb') as fo:
        result = pkl.load(fo, encoding='bytes')
    return {"data": result[b"data"], "labels": np.array(result[b"fine_labels"])}


def load_train_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(load_train_dataset_numpy())
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, count=FLAGS.num_epochs))
    dataset = dataset.batch(FLAGS.batch_size)
    def prepare_final_batch(x):
        batch_size = tf.shape(x["data"])[0]
        x["data"] = tf.cast(tf.reshape(x["data"], [batch_size, 3, 32, 32]), tf.float32) / 255.5
        x["data"] = tf.transpose(x["data"], [0, 2, 3, 1])
        x["labels"] = tf.cast(x["labels"], tf.int32)
        return x
    dataset = dataset.map(prepare_final_batch, num_parallel_calls=FLAGS.num_cores)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=FLAGS.num_cores))
    x = dataset.make_one_shot_iterator().get_next()
    return x


def load_test_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(load_test_dataset_numpy())
    dataset = dataset.batch(FLAGS.batch_size)
    def prepare_final_batch(x):
        batch_size = tf.shape(x["data"])[0]
        x["data"] = tf.cast(tf.reshape(x["data"], [batch_size, 32, 32, 3]), tf.float32)
        x["labels"] = tf.cast(x["labels"], tf.int32)
        return x
    dataset = dataset.map(prepare_final_batch, num_parallel_calls=FLAGS.num_cores)
    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=FLAGS.num_cores))
    x = dataset.make_one_shot_iterator().get_next()
    return x


class AttentionModule(tf.keras.layers.Layer):

    def __init__(self, num_heads, num_keys, num_values, num_outputs, name="attention_module", **kwargs):
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_outputs = num_outputs
        self.query_layer = tf.layers.Dense(
            self.num_keys * self.num_heads, 
            name=name + "/query_layer",
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.key_layer = tf.layers.Dense(
            self.num_keys * self.num_heads, 
            name=name + "/key_layer",
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.value_layer = tf.layers.Dense(
            self.num_values * self.num_heads, 
            name=name + "/value_layer",
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output_layer = tf.layers.Dense(
            self.num_outputs, 
            name=name + "/output_layer",
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        super(AttentionModule, self).__init__(name=name, **kwargs)
    
    def __call__(self, inputs):
        pre_queries = tf.expand_dims(tf.expand_dims(tf.reduce_mean(inputs, [1, 2]), 1), 1)
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        Q = tf.reshape(self.query_layer(pre_queries), [
            batch_size, 1, 1, self.num_heads, self.num_keys])
        K = tf.reshape(self.key_layer(inputs), [
            batch_size, height, width, self.num_heads, self.num_keys])
        V = tf.reshape(self.value_layer(inputs), [
            batch_size, height, width, self.num_heads, self.num_values])
        Q_horizontal = tf.transpose(Q, [0, 3, 1, 2, 4])
        K_horizontal = tf.transpose(K, [0, 3, 1, 2, 4])
        V_horizontal = tf.transpose(V, [0, 3, 1, 2, 4])
        Q_vertical = tf.transpose(Q, [0, 3, 2, 1, 4])
        K_vertical = tf.transpose(K, [0, 3, 2, 1, 4])
        V_vertical = tf.transpose(V, [0, 3, 2, 1, 4])
        A_horizontal = tf.matmul(tf.nn.softmax(tf.matmul(
            tf.tile(Q_horizontal, [1, 1, height, 1, 1]), 
            tf.transpose(K_horizontal, [0, 1, 2, 4, 3])) / tf.sqrt(float(self.num_keys))), V_horizontal)
        A_horizontal = tf.squeeze(A_horizontal, 3)
        A_horizontal = tf.transpose(A_horizontal, [0, 3, 2, 1])
        A_vertical = tf.matmul(tf.nn.softmax(tf.matmul(
            tf.tile(Q_vertical, [1, 1, width, 1, 1]), 
            tf.transpose(K_vertical, [0, 1, 2, 4, 3])) / tf.sqrt(float(self.num_keys))), V_vertical)
        A_vertical = tf.squeeze(A_vertical, 3)
        A_vertical = tf.transpose(A_vertical, [0, 3, 2, 1])
        A_result = tf.matmul(A_horizontal, tf.transpose(A_vertical, [0, 1, 3, 2]))
        A_result = tf.transpose(A_result, [0, 2, 3, 1])
        outputs = self.output_layer(A_result)  
        return outputs
        
    @property
    def trainable_variables(self):
        return (self.query_layer.trainable_variables + 
            self.key_layer.trainable_variables + 
            self.value_layer.trainable_variables + 
            self.output_layer.trainable_variables)
    
    @property
    def trainable_weights(self):
        return self.trainable_variables
    
    @property
    def variables(self):
        return (self.query_layer.variables + 
            self.key_layer.variables + 
            self.value_layer.variables + 
            self.output_layer.variables)
    
    @property
    def weights(self):
        return self.variables


if __name__ == "__main__":

    x = load_train_dataset()
    attention_module = AttentionModule(32, 3, 3, 3)
    y = attention_module(x["data"])
    loss = tf.nn.l2_loss(x["data"] - y)
    learning_step = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1"))) as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            sess.run(learning_step)
            if i % 100 == 0:
                print("learning step " + str(i))

        r = sess.run([x["data"], y])

        plt.imshow(r[0][0, ...])
        plt.savefig("before.png")
        plt.close()

        plt.imshow(r[1][0, ...])
        plt.savefig("after.png")
        plt.close()
