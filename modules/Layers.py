import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.distributions import MultivariateNormalDiag
import numpy as np
import tensorflow.contrib.rnn.python.ops.rnn_cell as rnn_cell
EPS=1e-5
class StochasticNet(object):
    def __init__(self,input_size,hidden_size,output_size, num_layers, nbatch,activation):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nbatch = nbatch
        self.activation=activation
        self.num_layers = []
        self.layers = []

        if num_layers == 1:
            l = SFFLayer(size=(input_size,output_size), nbatch=nbatch)
            self.layers.append(l)
        else:
            l = SFFLayer(size=(input_size,hidden_size), nbatch=nbatch, activation=activation)
            self.layers.append(l)
            for i in range(1,num_layers-1):
                l = SFFLayer(size=(hidden_size,hidden_size), nbatch=nbatch, activation=activation)
                self.layers.append(l)

            #final layer
            l = SFFLayer(size=(hidden_size,output_size), nbatch=nbatch, activation=None)
            self.layers.append(l)

    def forward(self,x, N=None):
        if N == None:
            N = len(self.layers)
        o = self.layers[0].sample(x)
        for i in range(1,N):
            o = self.layers[i].sample(o)

        return o

    def forward_dist(self,x):
        o = self.forward(x,self.num_layers-1)
        params = self.layers[-1].compute_params(o)
        return params

class SFFLayer(object):
    def __init__(self, size, nbatch, activation='tanh'):
        self.size = size
        self.weights_mu = tf.Variable(tf.random_normal(size, stddev=eps))
        self.bias_mu = tf.Variable(tf.zeros(size[1]))

        self.weights_sig = tf.Variable(tf.random_normal(size, stddev=eps))
        self.bias_sig = tf.Variable(tf.zeros(size[1]))
        self.activation = activation
        self.nbatch = nbatch

    def sample(self, x):
        mu,rho = self.compute_params(x)

        # reparameterization trick
        epsilon = tf.random_normal(shape=(self.nbatch,self.size[1]), mean=0., stddev=1.,
                                    dtype=tf.float32)
        self.reparam_outputs = mu + epsilon * rho
        if self.activation == 'relu':
            self.reparam_outputs = tf.nn.relu(self.reparam_outputs)
        elif self.activation == 'tanh':
            self.reparam_outputs = tf.nn.tanh(self.reparam_outputs)
        elif self.activation == 'sigmoid':
            self.reparam_outputs = tf.nn.sigmoid(self.reparam_outputs)

        return self.reparam_outputs

    def compute_params(self,x):
        mu = tf.matmul(x,self.weights_mu)+self.bias_mu
        rho = tf.matmul(x,self.weights_sig)+self.bias_sig
        rho = tf.log(1+tf.exp(rho)+EPS)
        return (mu,rho)

    def compute_prob(self,x,h):
        mu,rho = self.compute_params(x)

        return MultivariateNormalDiag(mu, tf.exp(rho), validate_args=False).pdf(h)
