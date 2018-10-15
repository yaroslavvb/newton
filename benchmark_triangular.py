#!/usr/bin/env/python
"""Benchmark triangular solve in numpy, Pytorch and TensorFlow.

On p3.2xlarge
Benchmarking n=8000
numpy
Times: min: 127.86, median: 128.12, mean: 128.31

tensorflow gpu
Times: min: 1.20, median: 1.20, mean: 1.20

tensorflow cpu
Times: min: 32.77, median: 32.88, mean: 32.91

pytorch cpu
Times: min: 162.65, median: 162.65, mean: 162.65

pytorch gpu
Times: min: 0.04, median: 0.05, mean: 0.05

"""

import scipy
import tensorflow as tf
tf.enable_eager_execution()

import torch
import util as u

if __name__ == '__main__':

  u.reset_timeit()

  iters = 3
  n = 8000

  print(f"Benchmarking n={n}")

  ############################################################
  # Numpy
  ############################################################
  A = scipy.randn(n, n)  # random matrix
  A = A @ A.T  # positive definite matrix
  A = scipy.linalg.cholesky(A)  # upper diagonal matrix
  b = scipy.randn(n)

  u.reset_timeit()
  for i in range(iters):
    with u.timeit('numpy'):
      scipy.linalg.solve_triangular(A, b)

  ############################################################
  # PyTorch
  ############################################################
  A = torch.randn(n, n)
  A = A @ A.t()+torch.diag(torch.ones(n))
  A = torch.potrf(A).cuda()
  b = torch.randn(n, 1).cuda()

  # prewarm
  torch.trtrs(b, A)
  for i in range(iters):
    with u.timeit('pytorch'):
      result = torch.trtrs(b, A)
    del result

  ############################################################
  # Tensorflow
  ############################################################
  A=tf.random_normal((n,n))
  b=tf.random_normal((n, 1))
  A=A@tf.transpose(A)+tf.diag(tf.ones((n,))) # bug, diag is needed, or Cholesky fails
  A=tf.cholesky(A)
  # bug, Should be able to do constant conversion, but fails with
  # Internal: failed to query device pointer for context: CUDA_ERROR_INVALID_VALUE
  #  A = tf.constant(A).gpu()
  #  b = tf.constant(b).gpu()
  #  assert 'gpu' in A.device.lower()

  # prewarm
  result =  tf.contrib.eager.Variable(tf.zeros((n, 1)))
  result.assign(tf.linalg.triangular_solve(A, b))
  for i in range(iters):
    b+=1  # prevent caching
    with u.timeit('tensorflow'):
      result.assign(tf.linalg.triangular_solve(A, b))
    print(result[0,0])

  u.summarize_timeit()
