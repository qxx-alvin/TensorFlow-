# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:09:15 2018

@author: Lenovo

自编码器
"""
# In[]
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt

# In[]
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = -low
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AGNAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function           # ???
        self.scale = tf.placeholder(tf.float32)     # ???
        self.training_scale = scale                 # ???
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                            self.x + self.scale * tf.random_normal((n_input,)),
                            
                            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.nn.softmax(tf.add(tf.matmul(self.hidden, 
                                        self.weights['w2']), self.weights['b2']))
        
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction, self.x), 2.0)) # reduce dimension???
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input],
                                                   dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
    
    # train one batch
    def partial_fit(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer),
                    feed_dict={self.x: X, self.scale: self.training_scale})
        return cost
    
    # one forward pass
    def calc_total_cost(self, X):
        return self.sess.run(self.cost,
                    feed_dict={self.x: X, self.scale: self.training_scale})
        
    # fetch encoding
    def transform(self, X):
        return self.sess.run(self.hidden,
                    feed_dict={self.x: X, self.scale: self.training_scale})
     
    # given hidden, calc output
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=len(self.weights["b1"]))  # ???
        return self.sess.run(self.reconstruction,
                    feed_dict={self.hidden: hidden})
    # fetch output    
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                    feed_dict={self.x: X, self.scale: self.training_scale})
        
    def getWeights(self):
        return self.sess.run(self.weights["w1"])        # ???
    
    def getBiases(self):
        return self.sess.run(self.weights["b1"])
# In[]
import pickle
with open('D:\\Documents\\Python\\mnist\\mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)
# In[]
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]

#X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
X_train, X_test = mnist.train.images, mnist.test.images

n_sample = int(mnist.train.num_examples)
training_epoch = 20
batch_size = 64
display_step = 1

autoencoder = AGNAutoEncoder(n_input=784,
                             n_hidden=300,
                             transfer_function=tf.nn.softplus,
                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                             scale=0.01)
# In[]
for epoch in range(training_epoch):
    avg_cost = 0.
    total_epoch = int(n_sample / batch_size)
    for i in range(total_epoch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        # avg_cost += cost / n_sample * batch_size
        avg_cost += cost
    
    avg_cost /= (total_epoch * batch_size)
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
# In[]
print("Avg cost: " + str(autoencoder.calc_total_cost(X_test) / len(X_test)))
# In[]
examples_to_show = 10
rec = autoencoder.reconstruct(mnist.test.images[:examples_to_show])
#rec = autoencoder.sess.run(autoencoder.reconstruction, feed_dict={
#                autoencoder.x: mnist.test.images[:examples_to_show],
#                autoencoder.scale: autoencoder.training_scale}) 
 
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):  
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  
    a[1][i].imshow(np.reshape(rec[i], (28, 28)))  
plt.show()  
# In[]
autoencoder.getWeights()