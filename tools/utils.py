from six.moves import cPickle as pickle
import numpy as np
import os
import platform


def load_pickle(f):
    """load python object"""
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
  """ load a batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_cifar10(mem_flag = True):
  """load cifar10 dataset
     Input
     mem_flag: flag to indicate if we are going to work with the whole dataset
               or just a small set
     Output
     Xtr: training data
     Ytr: training labels
     Xte: testing data
     Yte: testing labels
  """
  ROOT = "./cifar-10-batches-py"
  xs = []
  ys = []
  if mem_flag:
      end = 3
  else:
      end = 6
  for b in range(1,end):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  if mem_flag:
    return Xtr[:6000], Ytr[:6000], Xte[:600], Yte[:600]
  else:
    return Xtr, Ytr, Xte, Yte



