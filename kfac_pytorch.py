import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import scipy
import sys

from torch.autograd.function import Function

dsize = 10000
fs = [dsize, 28*28, 1024, 1024, 1024, 196, 1024, 1024, 1024, 28*28]
n = len(fs) - 2   # number of matmuls
covA_saved = [None]*n
covB_saved = [None]*n
A_saved =  [None]*n
Bhat_saved =  [None]*n
BB_saved =  [None]*n

def regularized_inverse(mat, lambda_=1e-3, inverse_method='numpy',
                        use_cuda=True):
#def regularized_inverse(mat, lambda_=3e-3, inverse_method='numpy',
#                        use_cuda=True):
  assert mat.shape[0] == mat.shape[1]
  ii = torch.eye(mat.shape[0])
  if use_cuda:
    ii = ii.cuda()
  regmat = mat + lambda_*ii

  if inverse_method == 'numpy':
    import util as u
    result = torch.from_numpy(scipy.linalg.inv(regmat.cpu().numpy()))
    if use_cuda:
      result = result.cuda()
  elif inverse_method == 'gpu':
    assert use_cuda
    result = torch.inverse(regmat).cuda()
  else:
    assert False, 'unknown inverse_method ' + str(INVERSE_METHOD)
  return result


def train(optimizer='sgd', nonlin=torch.sigmoid, kfac=True, iters=10,
          lr=0.2, newton_matrix='stochastic', eval_every_n_steps=1,
          print_interval=200, inverse_method='numpy'):
  """Train on first 10k MNIST examples, evaluate on second 10k."""

  global BB_saved
  
  u.reset_time()
  dsize = 10000

  # model options
  dtype = np.float32
  torch_dtype = 'torch.FloatTensor'

  use_cuda = torch.cuda.is_available()
  if use_cuda:
    torch_dtype = 'torch.cuda.FloatTensor'

  INVERSE_METHOD = 'numpy'  # numpy, gpu

  As = []
  Bs = []
  As_inv = []
  Bs_inv = []
  mode = 'capture'  # 'capture', 'kfac', 'standard'

  class KfacAddmm(Function):
    @staticmethod
    def _get_output(ctx, arg, inplace=False):
      if inplace:
        ctx.mark_dirty(arg)
        return arg
      else:
        return arg.new().resize_as_(arg)

    @staticmethod
    def forward(ctx, add_matrix, matrix1, matrix2, beta=1, alpha=1, inplace=False):
      ctx.save_for_backward(matrix1, matrix2)
      output = KfacAddmm._get_output(ctx, add_matrix, inplace=inplace)
      return torch.addmm(beta, add_matrix, alpha,
                         matrix1, matrix2, out=output)

    @staticmethod
    def backward(ctx, grad_output):
      matrix1, matrix2 = ctx.saved_variables
      grad_matrix1 = grad_matrix2 = None

      if mode == 'capture':
        Bs.insert(0, grad_output.data)
        As.insert(0, matrix2.data)
      elif mode == 'kfac':
        B = grad_output.data
        BB_saved.append(B.detach().cpu())
        A = matrix2.data
        Ainv = As_inv.pop()
        Binv = Bs_inv.pop()
        kfac_A = Ainv @ A
        kfac_B = Binv @ B
        grad_matrix1 = Variable(torch.mm(kfac_B, kfac_A.t()))
        #        print(Ainv[0,0].cpu().numpy(), Binv[0,0].cpu().numpy(), grad_matrix1[0,0].cpu().numpy())
      elif mode == 'standard':
        grad_matrix1 = torch.mm(grad_output, matrix2.t())

      else:
        assert False, 'unknown mode '+mode

      if ctx.needs_input_grad[2]:
        grad_matrix2 = torch.mm(matrix1.t(), grad_output)

      return None, grad_matrix1, grad_matrix2, None, None, None


  def kfac_matmul(mat1, mat2):
    output = Variable(mat1.data.new(mat1.data.size(0), mat2.data.size(1)))
    return KfacAddmm.apply(output, mat1, mat2, 0, 1, True)

  
  torch.manual_seed(1)
  np.random.seed(1)
  if use_cuda:
    torch.cuda.manual_seed(1)

  # feature sizes at each layer
  fs = [dsize, 28*28, 1024, 1024, 1024, 196, 1024, 1024, 1024, 28*28]
  n = len(fs) - 2   # number of matmuls

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      for i in range(1, n+1):
        W0 = u.ng_init(fs[i+1], fs[i])
        setattr(self, 'W'+str(i), nn.Parameter(torch.from_numpy(W0)))

    def forward(self, input):
      x = input.view(fs[1], -1)
      for i in range(1, n+1):
        W = getattr(self, 'W'+str(i))
        x = nonlin(kfac_matmul(W, x))
      return x.view_as(input)

  model = Net()

  if use_cuda:
    model.cuda()

  images = u.get_mnist_images()
  train_data0 = images[:, :dsize].astype(dtype)
  train_data = Variable(torch.from_numpy(train_data0))
  test_data0 = images[:, dsize:2*dsize].astype(dtype)
  test_data = Variable(torch.from_numpy(test_data0))
  if use_cuda:
    train_data = train_data.cuda()
    test_data = test_data.cuda()

  model.train()
  if optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
  elif optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
  else:
    assert False, 'unknown optimizer '+optimizer

  noise = torch.Tensor(*train_data.data.shape).type(torch_dtype)

  assert fs[-1]<=dsize
  padding = dsize-fs[-1]
  zero_mat = torch.zeros((fs[-1], padding))
  frozen = torch.cat([torch.eye(fs[-1]), zero_mat], 1).type(torch_dtype)

  covA_inv_saved = [None]*n
  losses = []
  vlosses = []
  
  for step in range(iters):
    mode = 'standard'
    output = model(train_data)

    if step>0:
      optimizer.step()

    if kfac:
      mode = 'capture'
      optimizer.zero_grad()
      del As[:], Bs[:], As_inv[:], Bs_inv[:]

      if newton_matrix == 'stochastic':
        noise.normal_()
        err_add = noise
      elif newton_matrix == 'exact':
        err_add = frozen
      else:
        assert False, 'unknown method for newton matrix '+newton_matrix

      output_hat = Variable(output.data+err_add)
      err_hat = output_hat - output
      loss_hat = torch.sum(err_hat*err_hat)/2/dsize
      loss_hat.backward(retain_graph=True)

      # compute inverses
      for i in range(n):
        # first layer activations don't change, only compute once
        if i == 0 and covA_inv_saved[i] is not None:
          covA_inv = covA_inv_saved[i]
        else:
          covA = As[i] @ As[i].t()/dsize
          covA_inv = regularized_inverse(covA,
                                         inverse_method=inverse_method)
          covA_inv_saved[i] = covA_inv
        As_inv.append(covA_inv)

        covA = As[i] @ As[i].t()/dsize
        covB = (Bs[i] @ Bs[i].t())*dsize
        covA_saved[i] = covA
        covB_saved[i] = covB
        
        A_saved[i] = As[i]
        Bhat_saved[i] = Bs[i]
        
        # alternative formula: slower but numerically better result
        # covB = (Bs[i]*dsize)@(Bs[i].t()*dsize)/dsize

        covB_inv = regularized_inverse(covB, inverse_method=inverse_method)
        Bs_inv.append(covB_inv)
      mode = 'kfac'
      BB_saved = []
      
    else:
      mode = 'standard'

    if step%eval_every_n_steps==0:
      old_mode = mode
      mode = 'standard'
      test_output = model(test_data)
      test_err = test_data - test_output
      test_loss = torch.sum(test_err*test_err)/2/dsize
      vloss0 = test_loss.data.cpu().numpy()
      vlosses.append(vloss0)
      mode = old_mode

    optimizer.zero_grad()
    err = output - train_data
    loss = torch.sum(err*err)/2/dsize
    loss.backward()
    
    loss0 = loss.data.cpu().numpy()
    losses.append(loss0)
    if step%print_interval==0:
      print("Step %3d loss %10.9f"%(step, loss0))

      
    u.record_time()

  return losses, vlosses, model, optimizer


def main():
  losses,vlosses = train(optimizer='sgd', kfac=True, nonlin=F.sigmoid, iters=10,
                 print_interval=1, lr=0.2)
  u.summarize_time()
  print(losses)
  loss0 = losses[-1]
  v = Variable('asdf')

  use_cuda = torch.cuda.is_available()
  if use_cuda:
    target = 38.781795502
  else:
    target = 0
  assert abs(loss0-target)<1e-9, abs(loss0-target)

if __name__=='__main__':
  main()
