#!/usr/bin/env/python
"""Benchmark triangular solve in numpy, Pytorch and TensorFlow.

On p3.2xlarge, DLAMI v15

Benchmarking n=10000
numpy
Times: min: 203.06, median: 203.48, mean: 204.37
Pytorch GPU
Times: min: 6.47, median: 6.53, mean: 6.64
Pytorch CPU
Times: min: 254.77, median: 263.34, mean: 268.56
TF GPU
Times: min: 11.65, median: 11.70, mean: 11.69
TF CPU
Times: min: 62.13, median: 63.75, mean: 63.86
"""

import scipy
import tensorflow as tf
tf.enable_eager_execution()

import torch
import util as u

def main():
  u.reset_timeit()

  iters = 11
  n = 10000

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
  # PyTorch GPU
  ############################################################
  A = torch.randn(n, n)
  A = A @ A.t()+torch.diag(torch.ones(n))
  A = torch.potrf(A).cuda()
  b = torch.randn(n, 1).cuda()

  # prewarm
  torch.trtrs(b, A)
  for i in range(iters):
    torch.cuda.synchronize()
    with u.timeit('Pytorch GPU'):
      result = torch.trtrs(b, A)
      torch.cuda.synchronize()
    del result

  ############################################################
  # PyTorch CPU
  ############################################################
  A = torch.randn(n, n)
  A = A @ A.t()+torch.diag(torch.ones(n))
  A = torch.potrf(A)
  b = torch.randn(n, 1)

  # prewarm
  (result, A_clone) = torch.trtrs(b, A)
  assert result.device.type == 'cpu'

  for i in range(iters):
    torch.cuda.synchronize()
    with u.timeit('Pytorch CPU'):
      result = torch.trtrs(b, A)
      torch.cuda.synchronize()
    del result

  ############################################################
  # PyTorch GPU
  ############################################################
  A = torch.randn(n, n)
  A = A @ A.t()+torch.diag(torch.ones(n))
  A = torch.potrf(A).cuda()
  b = torch.randn(n, 1).cuda()

  # prewarm
  (result, A_clone) = torch.trtrs(b, A)
  assert result.device.type == 'cuda'
  for i in range(iters):
    torch.cuda.synchronize()
    with u.timeit('Pytorch GPU'):
      (result, dummy) = torch.trtrs(b, A)
      print(result[0, 0])
      #      torch.cuda.synchronize()
    del result

  ############################################################
  # Tensorflow GPU
  ############################################################
  A=tf.random_normal((n,n)).gpu()
  b=tf.random_normal((n, 1)).gpu()
  A=A@tf.transpose(A)+tf.diag(tf.ones((n,))) # bug, diag is needed, or Cholesky fails
  A=tf.cholesky(A)
  # bug, Should be able to do constant conversion, but fails with
  # Internal: failed to query device pointer for context: CUDA_ERROR_INVALID_VALUE
  #  A = tf.constant(A).gpu()
  #  b = tf.constant(b).gpu()

  # prewarm
  result =  tf.contrib.eager.Variable(tf.zeros((n, 1)))
  result.assign(tf.linalg.triangular_solve(A, b))
  assert 'gpu' in result.device.lower()
  for i in range(iters):
    b+=1  # prevent caching
    with u.timeit('TF GPU'):
      result.assign(tf.linalg.triangular_solve(A, b))
      print(result[0, 0])

  ############################################################
  # Tensorflow CPU
  ############################################################
  A=tf.random_normal((n,n)).cpu()
  b=tf.random_normal((n, 1)).cpu()
  A=A@tf.transpose(A)+tf.diag(tf.ones((n,))) # bug, diag is needed, or Cholesky fails
  A=tf.cholesky(A)
  A = A.cpu()
  b = b.cpu()

  # prewarm
  with tf.device('/cpu:0'):
    result =  tf.contrib.eager.Variable(tf.zeros((n, 1)))
  result.assign(tf.linalg.triangular_solve(A, b))
  assert 'cpu' in result.device.lower()
  for i in range(iters):
    b+=1  # prevent caching
    with u.timeit('TF CPU'):
      result.assign(tf.linalg.triangular_solve(A, b))

  u.summarize_timeit()


if __name__ == '__main__':
  main()
