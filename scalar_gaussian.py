import numpy as np
import tensorflow as tf
from modules.Layers import StochasticNet
import matplotlib.pyplot as plt
import modules.utility as util
from scipy.stats import moment
EPS = 1e-5
############################
# Input parameters
############################
num_y_distributions = 3
output_dir = './{}'.format(num_y_distributions)
util.mkdir(output_dir)
input_dim = 1
output_dim = 1
hidden_size = 1
activation = 'tanh'
nbatch = 100
Ndata = 10000
Niter = 2000
N_mc = 20
print_step = 100
learning_rate = 1e-3
############################
# Generate Data
############################
r = np.random.rand(Ndata)
y = np.zeros((Ndata,output_dim))
x = np.zeros((Ndata,input_dim))

for i in range(num_y_distributions):
    dy = 1.0/(2*num_y_distributions-1)
    lbound = 2*i*dy
    ubound = (2*i+1)*dy
    inds = [j for j in range(Ndata) if (r[j]>=i*1.0/num_y_distributions and r[j] < (i+1)*1.0/num_y_distributions)]
    y[inds] = np.random.rand(len(inds),output_dim)*(ubound-lbound) + lbound

plt.figure()
plt.hist(y, bins = 30, label='y')
plt.savefig(output_dir+'/y_hist.png')
plt.close()

#train test split
y_test = y[int(-Ndata*0.25):]
x_test = x[int(-Ndata*0.25):]
y = y[:int(0.75*Ndata)]

############################
# Build networks
############################
net_1 = StochasticNet(input_size=input_dim,
    hidden_size=hidden_size,
    output_size=output_dim,
    num_layers=1,
    nbatch=nbatch,
    activation=activation)

net_2 = StochasticNet(input_size=input_dim,
    hidden_size=2,
    output_size=output_dim,
    num_layers=2,
    nbatch=nbatch,
    activation=activation)

net_3 = StochasticNet(input_size=input_dim,
    hidden_size=5,
    output_size=output_dim,
    num_layers=2,
    nbatch=nbatch,
    activation=activation)

#############################
# Set up tensorflow graph
#############################
x_tf = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
y_tf = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])

prob_1 = 0.0
prob_2 = 0.0
prob_3 = 0.0

for i in range(N_mc):
    prob_1 = prob_1+1.0/(N_mc)*net_1.forward_prob(x_tf,y_tf)
    prob_2 = prob_2+1.0/(N_mc)*net_2.forward_prob(x_tf,y_tf)
    prob_3 = prob_3+1.0/(N_mc)*net_3.forward_prob(x_tf,y_tf)

loss_1 = tf.reduce_mean(-tf.log(prob_1+EPS))
loss_2 = tf.reduce_mean(-tf.log(prob_2+EPS))
loss_3 = tf.reduce_mean(-tf.log(prob_3+EPS))

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

train_1 = optimizer.minimize(loss_1)
train_2 = optimizer.minimize(loss_2)
train_3 = optimizer.minimize(loss_3)

out_dict = {}
out_dict['loss_1'] = []
out_dict['loss_2'] = []
out_dict['loss_3'] = []

#############################
# Train
#############################
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(Niter):
    xb,yb = util.get_batch(x,y,nbatch)

    input_feed = {x_tf:xb,y_tf:yb}

    sess.run(train_1,input_feed)
    sess.run(train_2,input_feed)
    sess.run(train_3,input_feed)

    if i%print_step == 0:
        l1 = sess.run(loss_1,input_feed)
        l2 = sess.run(loss_2,input_feed)
        l3 = sess.run(loss_3,input_feed)

        out_dict['loss_1'].append(l1)
        out_dict['loss_2'].append(l2)
        out_dict['loss_3'].append(l3)

        print '{}: {},  {},  {}'.format(i, l1,l2,l3)

h_1 = net_1.forward(x_tf, size=x_test.shape[0])
h_2 = net_2.forward(x_tf, size=x_test.shape[0])
h_3 = net_3.forward(x_tf, size=x_test.shape[0])

yhat_1 = sess.run(h_1, {x_tf:x_test,y_tf:y_test})
yhat_2 = sess.run(h_2, {x_tf:x_test,y_tf:y_test})
yhat_3 = sess.run(h_3, {x_tf:x_test,y_tf:y_test})

plt.figure()
plt.hist(y_test, color='b',bins=50, label='test data', alpha=0.5)
plt.hist(yhat_1, color='r',bins=50, label='1 layer', alpha=0.5)
plt.hist(yhat_2, color='g',bins=50, label='2 layers, hidden 2', alpha=0.5)
plt.hist(yhat_3, color='k',bins=50, label='2 layers, hidden 3', alpha=0.5)
plt.legend()
plt.savefig(output_dir+'/out_hist.png')

plt.figure()
plt.plot(out_dict['loss_1'], color='r', label='1 layer', linewidth=2)
plt.plot(out_dict['loss_2'], color='g', label='2 layers, hidden 2', linewidth=2)
plt.plot(out_dict['loss_3'], color='k', label='2 layers, hidden 3', linewidth=2)
plt.legend()
plt.savefig(output_dir+'/train_loss.png')

#calculate moments
moms = [1,2,3,4,5,6,7,8,9]
ymoms = moment(y_test,moms)
ymoms_1 = moment(yhat_1,moms)
ymoms_2 = moment(yhat_2,moms)
ymoms_3 = moment(yhat_3,moms)

plt.figure()
plt.plot(ymoms, color='b', label='true data', linewidth=2)
plt.plot(ymoms_1, color='r', label='1 layer', linewidth=2)
plt.plot(ymoms_2, color='g', label='2 layers, hidden 2', linewidth=2)
plt.plot(ymoms_3, color='k', label='2 layers, hidden 3', linewidth=2)
plt.legend()
plt.savefig(output_dir+'/moments.png')
