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

train_x = all_x[:100].T
test_x = all_x[100:150].T

train_y = all_y[:100].T
test_y = all_y[100:150].T

#4x1
x = tf.placeholder(tf.float32, shape=[n_x, None])
#3x1
y = tf.placeholder(tf.float32, shape=[n_y, None])

#W = 4x3
W = tf.get_variable("weights", [n_y, n_x],dtype=tf.float32, initializer=tf.zeros_initializer)
#b = 1x3
b = tf.get_variable("bias", [n_y, 1],dtype=tf.float32, initializer=tf.zeros_initializer)

prediction = tf.nn.softmax(tf.matmul(W,x) - b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), axis=1))

gdo = tf.train.GradientDescentOptimizer(0.01)
optimizer = gdo.minimize(cost)

sess.run(tf.global_variables_initializer())

for epoch in range(10):
    sess.run([optimizer], feed_dict={x: train_x, y: train_y})

print(test_x.values[:,0])
changed_x = test_x.values[:,0]
changed_x.shape = (4,None)
print(changed_x.shape)

print(sess.run(prediction, feed_dict={x: test_x.values[:,0].T, y: test_y.values[:,0].T}).tolist()[0])




