import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

def get_batch(xdata, ydata, nbatch):
    N = len(ydata)
    inds = np.random.choice(N, size=nbatch, replace=True)
    xret = xdata[inds,:]
    if len(ydata.shape) == 1:
        yret = ydata[inds]
    else:
        yret = ydata[inds,:]
    return (xret,yret)


def data_iter(data, batch_size):
    n_batches = data.images.shape[0] / batch_size
    x_batches = np.array_split(data.images, n_batches)
    y_batches = np.array_split(data.labels, n_batches)
    batches = [(x, y) for (x, y) in zip(x_batches, y_batches)]
    np.random.shuffle(batches)
    for x, y in batches:
        yield (x, y)

def read_mnist(save_dir):
    mnist = input_data.read_data_sets(save_dir, one_hot=True)
    return mnist.train, mnist.validation, mnist.test

def mkdir(fn):
    if not os.path.exists(os.path.abspath(fn)):
        os.mkdir(os.path.abspath(fn))

if __name__ == '__main__':
    train, val, test = read_mnist('./data/')
    for m, l in data_iter(train, 100):
        print m.shape
        print l.shape
