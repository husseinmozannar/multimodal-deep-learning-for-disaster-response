from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import random
import pickle
#from representationsLinear import batchloadData


def deepnn(x):


  # Fully connected layer 1 
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([12, 6])
    b_fc1 = bias_variable([6])
    h_pool1_flat = tf.reshape(x, [-1,12]) # might be useless
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout1'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
  '''
  # Fully connected layer 2
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([10, 6])
    b_fc2 = bias_variable([6])
    h_pool2_flat = tf.reshape(h_fc1, [-1,10])
    h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)
  
  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([6, 6])
    b_fc3 = bias_variable([6])
    h_pool3_flat = tf.reshape(h_fc2, [-1,6])
    h_fc3 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc3) + b_fc3)
  
  with tf.name_scope('fc4'):
    W_fc4 = weight_variable([32, 16])
    b_fc4 = bias_variable([16])
    h_pool4_flat = tf.reshape(h_fc3, [-1,32])
    h_fc4 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc4) + b_fc4)
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.

  #with tf.name_scope('dropout2'):
  #  h_fc2 = tf.nn.dropout(h_fc2, keep_prob)
  # Fully connected layer 3
  
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.

  '''
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  # Map the 256 features to 3 classes
  with tf.name_scope('end'):
    W_fc_end = weight_variable([6, 6])
    b_fc_end = bias_variable([6])
    y_conv = tf.matmul(h_fc1, W_fc_end) + b_fc_end
    
  return y_conv, keep_prob



def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def nnClassifier():

  #load the data
  with open('image_decision_cv3_train.pkl', 'rb') as f: 
      x_train_image, y_train = pickle.load(f)

  with open('image_decision_cv3_test.pkl','rb') as f:  
      x_test_image, y_test = pickle.load(f)

  with open('image_decision_cv3_val.pkl','rb') as f: 
      x_val_image, y_val = pickle.load(f)

  print("loaded images")
  with open('text_decision_cv3_train.pkl', 'rb') as f:  
      x_train_text = pickle.load(f)

  with open('text_decision_cv3_test.pkl','rb') as f:  
      x_test_text = pickle.load(f)

  with open('text_decision_cv3_val.pkl','rb') as f:  
      x_val_text = pickle.load(f)

  print("loaded text")
  print("arranging data")
  #arrange data
  x_train = []
  x_test = []
  x_val = []
  
  for i in range(0,len(x_test_image)):
    joint = np.concatenate([x_test_image[i],x_test_text[i]])
    x_test = np.concatenate([x_test,joint])
    
  for i in range(0,len(x_train_image)):
    joint = np.concatenate([x_train_image[i],x_train_text[i]])
    x_train = np.concatenate([x_train,joint])
         
  for i in range(0,len(x_val_image)):
    joint = np.concatenate([x_val_image[i],x_val_text[i]])
    x_val = np.concatenate([x_val,joint])

  x_val = x_val.reshape(len(x_val_image),12)
  x_test = x_test.reshape(len(x_test_image),12)
  x_train = x_train.reshape(len(x_train_image),12)


  # Create the model
  x = tf.placeholder(tf.float32, [None, 12])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 6])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)


  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    
  cross_entropy = tf.reduce_mean(cross_entropy)
  
  with tf.name_scope('adam_optimizer'):
    step = tf.Variable(0,trainable=False)
    starter_learning_rate = 0.01
    end_learning_rate = 0.0001
    decay_steps = 20000
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, step,
                                          decay_steps, end_learning_rate,
                                          power=1.0)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=step)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)


  batch_size= int(x_train.shape[0] /100 +1 )
  
  #randomly shuffle x and y
  temp_list = list(zip(x_train,y_train))
  random.shuffle(temp_list)
  x_train ,y_train =zip(*temp_list)
  
  #split x and y into batches of size ~50
  x_batches = np.array_split(x_train,batch_size)
  y_batches = np.array_split(y_train,batch_size)
  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
  print("Starting training")              
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
      for j in range(len(x_batches)):
        train_step.run(feed_dict={x: x_val, y_: y_val, keep_prob: 0.9})
        train_step.run(feed_dict={x: x_batches[j], y_: y_batches[j], keep_prob: 0.5})

      if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x: x_train, y_: y_train, keep_prob: 1.0})
        print('step %d, training accuracy %.5f %%' % (i, train_accuracy*100))
        
        train_loss = cross_entropy.eval(feed_dict ={
          x: x_train, y_: y_train ,keep_prob: 1.0})
        print('step %d, cross entropy loss %.10e' % (i, train_loss))
      if i % 100 == 0:
        val_accuracy = accuracy.eval(feed_dict={
            x: x_val, y_: y_val, keep_prob: 1.0})
        print('step %d, validation accuracy %.5f %%' % (i, val_accuracy*100))


    print('Test accuracy %f' % accuracy.eval(feed_dict={
        x: x_test, y_: y_test, keep_prob: 1.0}))


