# -*- coding: utf-8 -*-
"""code_lab3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_mY54zcBE7uEpQF9JgOUpAwTyrn1afGg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import collections

from google.colab import drive
drive.mount('/content/gdrive')

open('/content/gdrive/My Drive/LAB/train-data.csv').read()

#Loading subsets
train_set = np.loadtxt("/content/gdrive/My Drive/LAB/train-data.csv", delimiter = ',')
test_set = np.loadtxt("/content/gdrive/My Drive/LAB/test-data.csv", delimiter = ',')
#Loading targets
train_targets = pd.read_csv("/content/gdrive/My Drive/LAB/train-target.csv", sep = ',', header = None)
test_targets = pd.read_csv("/content/gdrive/My Drive/LAB/train-target.csv", sep = ',', header = None)

print(train_set)
print(test_set)

#Labels binarizer
lb = preprocessing.LabelBinarizer()
lb.fit(train_targets[0])
target_train_matrix = lb.transform(train_targets)
target_test_matrix = lb.transform(test_targets)

target_train_matrix.shape

#Let's print one of the characters contained in the training set
x = train_set[0].reshape(16,8)
plt.gray()
plt.matshow(1.0 - x)
plt.show()
plt.close()

#Placeholders are variables without data assigned yet
x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 26]))
b = tf.Variable(tf.zeros([26]))
y = tf.placeholder(tf.float32, [None, 26])
y_hat = tf.nn.softmax(tf.matmul(x, W)+b)

def next_batch(num, data, labels):
  idx = np.arange(0, len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Initialization of weights
tf.set_random_seed(1234)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Initialization of Bias
def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME', use_cudnn_on_gpu = False)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides  = [1,1,1,1], padding = 'SAME', use_cudnn_on_gpu = False)

#Structure
x_image = tf.reshape(x, [-1,16,8,1])

#1st CONVOLUTIONAL LAYER
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd CONVOLUTIONAL LAYER
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Flat layer
W_fc1 = weight_variable([4*2*64, 512])
b_fc1 = bias_variable([512])

#Max pool for the flat layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

#Dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Output dense node
W_fc2 = weight_variable([512, 26])
b_fc2 = bias_variable([26])

#Output nodes
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices = [1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

predicted_y = tf.argmax(y_hat, 1)
real_y = tf.argmax(y, 1)
correct_prediction = tf.equal(predicted_y, real_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size_train = 20
epochs = train_set.shape[0]//batch_size_train
sess = tf.InteractiveSession()
print(epochs, train_set.shape[0])

#TRAINING AND TESTING 1
sess.run(tf.global_variables_initializer())
for i in range(epochs):
  batch = next_batch(batch_size_train, train_set, target_train_matrix)
  if i % 100 == 0:
    print(type(batch[1]))
    train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})
    print("Step {}, Training Accuracy {}".format(i, train_accuracy))

  train_step.run(feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5})

accuracy_values = []
predicted_targets = []

batch_size_test = 61
for j in range(9*19):
  batch = next_batch(batch_size_test, test_set, target_test_matrix)
  b_acc, b_pred_y = sess.run([accuracy, predicted_y], feed_dict ={x: batch[0], y: batch[1], keep_prob: 1.0})

  accuracy_values.append(b_acc)
  predicted_targets.extend(b_pred_y)

print('test accuracy {}'.format(np.mean(accuracy_values)))

#Cross validation over the rate learning
rates = [1e-3, 5e-4, 1e-4, 6e-4]

from sklearn.model_selection import KFold

kf = KFold(n_splits = 4)

#Algorithm
cv_scores = []

for rate in rates:
  print("Cross Validation for parameter rate = {}".format(rate))
  acc_list = []
  train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)

  for train, val in kf.split(train_set):
    CV_EPOCHS = epochs
    sess.run(tf.global_variables_initializer())
    t_data = train_set[train]
    v_data = train_set[val]
    t_labels = target_train_matrix[train]
    v_labels = target_train_matrix[val]

    for i in range(CV_EPOCHS):
      j, k = i*batch_size_train, (i+1)*batch_size_train
      train_step.run(feed_dict = {x: t_data[j:k], y: t_labels[j:k], keep_prob: 0.5})

    acc = sess.run(accuracy, feed_dict = {x: v_data, y: v_labels, keep_prob: 1.0})
    acc_list.append(acc)
    print("CV accuracy: {}".format(acc))

  cv_scores.append(acc_list)

cv_scores = np.array(cv_scores)
np.savetxt("/content/gdrive/My Drive/LAB/cv_scores.txt", cv_scores)

mean_cv_acc = np.apply_along_axis(np.mean, 1, cv_scores)
best_rate = rates[mean_cv_acc.argmax()]
best_rate

print(mean_cv_acc[0], mean_cv_acc[1], mean_cv_acc[2], mean_cv_acc[3])

#TRAINING AND TESTING USING THE BEST LEARNING RATE
train_step = tf.train.AdamOptimizer(best_rate).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
for i in range(epochs):
    batch = next_batch(batch_size_train,train_set,target_train_matrix)
    if i % 100 == 0:
        print(type(batch[1]))
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], 
                       y: batch[1], 
                       keep_prob: 1.0})
        print('step {}, training accuracy {}'.format(i, 
                                                     train_accuracy))
    train_step.run(feed_dict={x: batch[0], 
                              y: batch[1], 
                              keep_prob: 0.5})
    
accuracy_values = []
predicted_targets = []

batch_size_test = 61
for j in range(9*19):
  batch = next_batch(batch_size_test, test_set, target_test_matrix)
  b_acc, b_pred_y = sess.run([accuracy, predicted_y], feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})

  accuracy_values.append(b_acc)
  predicted_targets.extend(b_pred_y)

print("Test Accuracy {}".format(np.mean(accuracy_values)))

#Back to readable format
from string import ascii_lowercase
pred_letters = [ascii_lowercase[i] for i in predicted_targets]
pd.DataFrame(pred_letters).to_csv("/content/gdrive/My Drive/LAB/final_predictions.csv", header = False, index = False)
pd.DataFrame(pred_letters).to_csv("/content/gdrive/My Drive/LAB/final_predictions_csv.txt", header = False, index = False)

