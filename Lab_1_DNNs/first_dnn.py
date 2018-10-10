############################################################
#                                                          #
#  Code for Lab 1: Your First Fully Connected Layer  #
#                                                          #
############################################################


import tensorflow as tf
import os
import os.path
import numpy as np
import pandas as pd

sess = tf.Session()

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", sep=",",
                   names=["sepal_length", "sepal_width", "petal_length", "petal_width", "iris_class"])
#

np.random.seed(0)
data = data.sample(frac=1).reset_index(drop=True)
#
all_x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
#
all_y = pd.get_dummies(data.iris_class)
#

n_x = len(all_x.columns)
n_y = len(all_y.columns)

train_x = all_x[:100]
test_x = all_x[100:150]

train_y = all_y[:100]
test_y = all_y[100:150]

#4x1
x = tf.placeholder(tf.float32, shape=[None, n_x])
#3x1
y = tf.placeholder(tf.float32, shape=[None, n_y])

#W = 4x3
W = tf.get_variable("weights", [n_x, n_y],dtype=tf.float32, initializer=tf.zeros_initializer)
#b = 1x3
b = tf.get_variable("bias", [1, n_y],dtype=tf.float32, initializer=tf.zeros_initializer)

prediction = tf.nn.softmax(tf.matmul(x,W) - b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))

gdo = tf.train.GradientDescentOptimizer(0.01)
optimizer = gdo.minimize(cost)

sess.run(tf.global_variables_initializer())

tf_metric, tf_metric_update = tf.metrics.accuracy(tf_label, tf_prediction, name="my_metric")

# Isolate the variables stored behind the scenes by the metric operation
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

# Define initializer to initialize/reset running variables
running_vars_initializer = tf.variables_initializer(var_list=running_vars)

sess.run(tf.running_vars_initializer)
for i in range(1):
    for epoch in range(10):
        sess.run([optimizer], feed_dict={x: train_x, y: train_y})
        predict_values = tf.argmax(sess.run(prediction, feed_dict={x: train_x, y: train_y}), 1)
        label_values = tf.argmax(train_y, 1)
        acc, acc_op = tf.metrics.accuracy(labels=label_values, predictions=predict_values)
        sess.run([acc, acc_op])
    print("Accuracy at epoch " + str(i*epoch) + " = " + str(sess.run([acc])[0]))