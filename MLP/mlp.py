# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 10:57:17 2018

@author: Administrator

多层神经网络，采用Dropout、Adagrad、ReLU解决MNIST分类
"""

# In[]
import pickle
with open('D:\\Documents\\Python\\mnist\\mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)
    
import tensorflow as tf
sess = tf.InteractiveSession()

# In[]
in_units = 784
hidden_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, hidden_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_units]))
W2 = tf.Variable(tf.zeros([hidden_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden_drop = tf.nn.dropout(hidden, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(.3).minimize(cross_entropy)

tf.global_variables_initializer().run()
for i in range(3000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x: batch_x, y_: batch_y, keep_prob: 0.75})
# In[]
correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    