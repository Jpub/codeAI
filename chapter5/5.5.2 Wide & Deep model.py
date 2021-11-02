#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf

class WideAndDeepModel:
    def __init__(self, wide_length, deep_length, deep_last_layer_len, softmax_label):
        self.input_wide_part = tf.placeholder(tf.float32, shape=[None, wide_length], name='input_wide_part')
        self.input_deep_part = tf.placeholder(tf.float32, shape=[None, deep_length], name='input_deep_part')
        self.input_y = tf.placeholder(tf.float32, shape=[None, softmax_label], name='input_y')

        with tf.name_scope('deep_part'):
            w_x1 = tf.Variable(tf.random_normal([wide_length, 64], stddev=0.03), name='w_x1')
            b_x1 = tf.Variable(tf.random_normal([64]), name='b_x1')

            w_x2 = tf.Variable(tf.random_normal([64, deep_last_layer_len], stddev=0.03), name='w_x2')
            b_x2 = tf.Variable(tf.random_normal([deep_last_layer_len]), name='b_x2')

            z1 = tf.add(tf.matmul(self.input_wide_part, w_x1), b_x1)
            a1 = tf.nn.relu(z1)
            self.deep_logits = tf.add(tf.matmul(a1, w_x2), b_x2)

        with tf.name_scope('wide_part'):
            weights = tf.Variable(tf.truncated_normal([deep_last_layer_len + wide_length, softmax_label]))
            biases = tf.Variable(tf.zeros([softmax_label]))

            self.wide_and_deep = tf.concat([self.deep_logits, self.input_wide_part], axis = 1)

            self.wide_and_deep_logits = tf.add(tf.matmul(self.wide_and_deep, weights), biases)
            self.predictions = tf.argmax(self.wide_and_deep_logits, 1, name= "prediction")


        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.wide_and_deep_logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


# In[ ]:


# 데이터와 레이블 읽어오기
import pandas as pd
import numpy as np
import csv

def load_data_and_labels(path):
    data = []
    y = []
    total_q = []

    # count = 0
    with open(path, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        for row in rdr:
            y.append(float(row[1]))


    # data = np.asarray(data)
    total_q = np.asarray(total_q)
    y = np.asarray(y)
    return data, y


data, y = load_data_and_labels('../data/zutao2.csv')

bins = pd.qcut(y, 50, retbins=True)
print(bins[0])


# In[ ]:


import tensorflow as tf
import data_helpers
import os
import datetime
import time
from WideandDeepModel import WideAndDeepModel

# Data loading params
tf.flags.DEFINE_string("train_dir", "../data/cvr_train_data.csv", "Path of train data")
tf.flags.DEFINE_integer("wide_length", 261, "Path of train data")
tf.flags.DEFINE_integer("deep_length", 261, "Path of train data")
tf.flags.DEFINE_integer("deep_last_layer_len", 32, "Path of train data")
tf.flags.DEFINE_integer("softmax_label", 1, "Path of train data")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs")
tf.flags.DEFINE_integer("display_every", 50, "Number of iterations to display training info.")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with.")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps")



# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def train():
    with tf.device('/cpu:0'):
        x, y = data_helpers.load_data_and_labels(FLAGS.train_dir)

    print('-' * 120)
    print(x.shape)
    print(y.shape)
    print('-' * 120)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            model = WideAndDeepModel(
                wide_length=FLAGS.wide_length,
                deep_length=FLAGS.deep_length,
                deep_last_layer_len=FLAGS.deep_last_layer_len,
                softmax_label=FLAGS.softmax_label
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            #
            # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_dir = '/Users/asukapan/workspace/all_codes/iscp_all_codes/src/wide_and_deep_for_cvr/src/model/'
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                feed_dict = {
                    model.input_wide_part: x_batch,
                    model.input_deep_part: x_batch,
                    model.input_y: y_batch
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, model.loss, model.accuracy], feed_dict)

                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, auc {:G}".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

        save_path = saver.save(sess, checkpoint_prefix)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()

