############################################################
#                                                          #
#  Code for Lab 1: Intro to TensorFlow and Blue Crystal 4  #
#                                                          #
############################################################

'''Based on TensorFLow's tutorial: A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import os.path

import tensorflow as tf
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CIFAR10'))
import cifar10 as cf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', os.getcwd() + '/dataset/',
                            'Directory where the dataset will be stored and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'Number of steps between logging results to the console and saving summaries (default: %(default)d)')
tf.app.flags.DEFINE_integer('save_model', 1000,
                            'Number of steps between model saves (default: %(default)d)')

# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 256, 'Number of examples per mini-batch (default: %(default)d)')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_width', 32, 'Image width (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_height', 32, 'Image height (default: %(default)d)')
tf.app.flags.DEFINE_integer('img_channels', 3, 'Image channels (default: %(default)d)')
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes (default: %(default)d)')
tf.app.flags.DEFINE_string('log_dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'Decay steps for learning rate')
tf.app.flags.DEFINE_float('decay_rate', 0.8, 'Decay rate for learning rate')

run_log_dir = os.path.join(FLAGS.log_dir,
                           'exp_BN_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                        lr=FLAGS.learning_rate))

# run_log_dir = os.path.join(FLAGS.log_dir,
#                            'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
#                                                         lr=FLAGS.learning_rate))


xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial, name='weights')
    return tf.Variable(xavier_initializer(shape), name='weights')

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial, name='biases')
    return tf.Variable(xavier_initializer(shape), name='biases')

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

  Args:
      x: an input tensor with the dimensions (N_examples, 3072), where 3072 is the
        number of pixels in a standard CIFAR10 image.

  Returns:
      y: is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the object images into one of 10 classes
        (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
      img_summary: a string tensor containing sampled input images.
    """ 
    # Reshape to use within a convolutional neural net.  Last dimension is for
    # 'features' - it would be 1 one for a grayscale image, 3 for an RGB image,
    # 4 for RGBA, etc.

    x_image = tf.reshape(x, [-1, FLAGS.img_width, FLAGS.img_height, FLAGS.img_channels])

    img_summary = tf.summary.image('Input_images', x_image)

    # # First convolutional layer - maps one image to 32 feature maps.
    # with tf.variable_scope('Conv_1'):
    W_conv1 = weight_variable([5, 5, FLAGS.img_channels, 32])
    b_conv1 = bias_variable([32])
    # h_conv1_bn = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv1)

    Z1 = tf.nn.conv2d(x_image, W_conv1, strides=[
                      1, 1, 1, 1], padding='SAME', name='convolution') + b_conv1
    Z1_mean, Z1_variance = tf.nn.moments(Z1, [0,1,2])
    Z1_standard_dev = tf.sqrt(Z1_variance)

    if Z1_standard_dev == 0:
      Z1_standard_dev = 0.000001

    Z1_hat = (Z1 - Z1_mean) / Z1_standard_dev

    gamma1 = weight_variable([ 32, 32, 32])
    beta1 = bias_variable([32])

    B1 = Z1_hat * gamma1 + beta1
    h_conv1_bn = tf.nn.relu(B1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = tf.nn.max_pool(h_conv1_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')
        
    with tf.variable_scope('Conv_2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2_bn = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME', name='convolution') + b_conv2)

        Z2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[
                      1, 1, 1, 1], padding='SAME', name='convolution') + b_conv2
        Z2_mean, Z2_variance = tf.nn.moments(Z2, [0,1,2])
        Z2_standard_dev = tf.sqrt(Z2_variance)

        if Z2_standard_dev == 0:
            Z2_standard_dev = 0.000001

        Z2_hat = (Z2 - Z2_mean) / Z2_standard_dev

        gamma2 = weight_variable([16, 16, 64])
        beta2 = bias_variable([64])

        B2 = gamma2 * Z2_hat + beta2
        h_conv2_bn = tf.nn.relu(B2)

        # Pooling layer - downsamples by 2X.
        h_pool2 = tf.nn.max_pool(h_conv2_bn, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name='pooling')
                          
        # You need to continue building your convolutional network!
    with tf.variable_scope('Conv_out'):
        conv_out = tf.reshape(h_pool2, [-1,4096])

    with tf.variable_scope('FCN_1'):
        W_fcn1 = weight_variable([4096, 1024])
        b_fcn1 = bias_variable([1024])
        h_fcn1 = tf.nn.relu(tf.matmul(conv_out, W_fcn1) + b_fcn1)

    with tf.variable_scope('FCN_2'):
        W_fcn2 = weight_variable([1024, 1024])
        b_fcn2 = bias_variable([1024])
        h_fcn2 = tf.nn.relu(tf.matmul(h_fcn1, W_fcn2) + b_fcn2)

    with tf.variable_scope('Out'):
        W_out = weight_variable([1024, FLAGS.num_classes])
        b_out = bias_variable([FLAGS.num_classes])
        y_conv = tf.matmul(h_fcn2, W_out) + b_out
        
    return y_conv, img_summary


def main(_):
    tf.reset_default_graph()

    # Import data
    cifar = cf.cifar10(batchSize=FLAGS.batch_size, downloadDir=FLAGS.data_dir)

    with tf.variable_scope('inputs'):
        # Create the model
        x = tf.placeholder(tf.float32, [None, FLAGS.img_width * FLAGS.img_height * FLAGS.img_channels])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

    # Build the graph for the deep net
    y_conv, img_summary = deepnn(x)
    # Define your loss function - softmax_cross_entropy
    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    # Define your AdamOptimiser, using FLAGS.learning_rate to minimixe the loss function
    
    # adam = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # optimizer = adam.minimize(cross_entropy)

    global_step = tf.Variable(0, trainable=False)
    decay_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           FLAGS.decay_steps, FLAGS.decay_rate)
    # Passing global_step to minimize() will increment it at each step.
    optimizer = (
     tf.train.AdamOptimizer(decay_learning_rate)
     .minimize(cross_entropy, global_step=global_step)
    )

    # calculate the prediction and the accuracy


    correct_prediction = tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.float32)
    
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(correct_prediction)
    
    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Accuracy', accuracy)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(run_log_dir + '_train', sess.graph)
        summary_writer_validation = tf.summary.FileWriter(run_log_dir + '_validate', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            # Training: Backpropagation using train set
            (trainImages, trainLabels) = cifar.getTrainBatch()
            (testImages, testLabels) = cifar.getTestBatch()
            
            _, summary_str = sess.run([optimizer, training_summary], feed_dict={x: trainImages, y_: trainLabels})

            
            if step % (FLAGS.log_frequency + 1)== 0:
               summary_writer.add_summary(summary_str, step)

            #Validation: Monitoring accuracy using validation set
            if step % FLAGS.log_frequency == 0:
               validation_accuracy, summary_str = sess.run([accuracy, validation_summary], feed_dict={x: testImages, y_: testLabels})
               print('step %d, accuracy on validation batch: %g' % (step, validation_accuracy))
               summary_writer_validation.add_summary(summary_str, step)

            #Save the model checkpoint periodically.
            if step % FLAGS.save_model == 0 or (step + 1) == FLAGS.max_steps:
               checkpoint_path = os.path.join(run_log_dir + '_train', 'model.ckpt')
               saver.save(sess, checkpoint_path, global_step=step)

        # Testing

        # resetting the internal batch indexes
        cifar.reset()
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        # don't loop back when we reach the end of the test set
        while evaluated_images != cifar.nTestSamples:
            (testImages, testLabels) = cifar.getTestBatch(allowSmallerBatches=True)
            test_accuracy_temp, _ = sess.run([accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

            batch_count = batch_count + 1
            test_accuracy = test_accuracy + test_accuracy_temp
            evaluated_images = evaluated_images + testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)



if __name__ == '__main__':
    tf.app.run(main=main)
