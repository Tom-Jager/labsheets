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

g = tf.get_default_graph()
with g.as_default():

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

  for epoch in range(1000):
    sess.run([optimizer], feed_dict={x: train_x, y: train_y})



  h1 = 10
  h2 = 20
  h3 = 10

  W_fc1 = tf.Variable(tf.truncated_normal([n_x, h1], stddev=0.1))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[h1]))
  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  W_fc2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[h2]))
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  W_fc3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.1))
  b_fc3 = tf.Variable(tf.constant(0.1, shape=[h3]))
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

  W_ol = tf.Variable(tf.truncated_normal([h3, n_y], stddev=0.1))
  b_ol = tf.Variable(tf.constant(0.1, shape=[n_y]))
  predictions_fcn = tf.nn.relu(tf.matmul(h_fc3, W_ol) + b_ol)

  with tf.name_scope('loss'):
    cost_fcn = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=predictions_fcn, scope="Cost_Function")
    tf.summary.scalar('loss', cost_fcn)

  adagrad = tf.train.AdagradOptimizer(0.1)
  optimizer_fcn = adagrad.minimize(cost_fcn)

  prediction_correct = tf.cast(tf.equal(tf.argmax(predictions_fcn,1), tf.argmax(y,1)), tf.float32)

  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(prediction_correct)
    tf.summary.scalar('accuracy', accuracy)

  sess.run(tf.global_variables_initializer())
  
  logs_path = "./logs/"
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(logs_path + '/train')
  test_writer = tf.summary.FileWriter(logs_path + '/test')

  for epoch in range(3000):
    train_summary, _ = sess.run([merged, optimizer_fcn], feed_dict={x: train_x, y: train_y})
    if (epoch % 100 == 0):
      test_summary, acc = sess.run([merged, accuracy], feed_dict={x: test_x, y: test_y})

    train_writer.add_summary(train_summary, epoch)
    test_writer.add_summary(test_summary, epoch)
    print("Accuracy at epoch:" + str(epoch) + " is " + str(acc))

  
  





