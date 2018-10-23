import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd

import util as u

import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--run', type=str, default='psgd-lenet',
                    help='name of run')
parser.add_argument('--test', action='store_true',
                    help='simple numeric test')
args = parser.parse_args()

num_updates = 5


# class LeNet(nn.Module):
#   def __init__(self):
#     super(LeNet, self).__init__()
#     self.conv1 = nn.Conv2d(1, 6, 5)
#     self.conv2 = nn.Conv2d(6, 16, 5)
#     self.fc1 = nn.Linear(16 * 4 * 4, 120)
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 10)

#   def forward(self, x):
#     x = F.relu(self.conv1(x))
#     x = F.max_pool2d(x, 2)
#     x = F.relu(self.conv2(x))
#     x = F.max_pool2d(x, 2)
#     num_outputs = np.prod(x.shape)
#     batch_size = x.shape[0]
#     x = x.view(-1, num_outputs // batch_size)
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     x = F.log_softmax(x, dim=1)
#     return x


class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5, self).__init__()
    W0 = torch.tensor(0.1 * torch.randn(1 * 5 * 5 + 1, 6), requires_grad=True)
    W1 = torch.tensor(0.1 * torch.randn(6 * 5 * 5 + 1, 16), requires_grad=True)
    W2 = torch.tensor(0.1 * torch.randn(16 * 4 * 4 + 1, 120), requires_grad=True)  # so here is 4x4, not 5x5
    W3 = torch.tensor(0.1 * torch.randn(120 + 1, 84), requires_grad=True)
    W4 = torch.tensor(0.1 * torch.randn(84 + 1, 10), requires_grad=True)
    self.W = [W0, W1, W2, W3, W4]

  def to(self, device):
    for i in range(len(self.W)):
      self.W[i] = self.W[i].to(device)
    return self

  def forward(self, x):
    W = self.W
    x = F.conv2d(x, W[0][:-1].view(6, 1, 5, 5), bias=W[0][-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.conv2d(x, W[1][:-1].view(16, 6, 5, 5), bias=W[1][-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.relu(x.view(-1, 16 * 4 * 4).mm(W[2][:-1]) + W[2][-1])
    x = F.relu(x.mm(W[3][:-1]) + W[3][-1])
    x = x.mm(W[4][:-1]) + W[4][-1]
    return x

  def loss(self, data, target):
    y = self.forward(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    for w in self.W:
      loss += 0.0002 * torch.sum(w * w)

    return loss


def loopy(dl):
  while True:
    yield from iter(dl)


class Preconditioner:
  def __init__(self, net, loader, device):
    self.net = net
    self.loader = loader
    self.iter = loopy(loader)  # non-stop dataloader
    self.device = device
    Qs = [[torch.eye(w.shape[0]), torch.eye(w.shape[1])] for w in net.W]
    for i in range(len(Qs)):
      for j in range(len(Qs[i])):
        Qs[i][j] = Qs[i][j].to(device)
    self.Qs = Qs
    self.step_count = 0
    self.logger = u.get_last_logger()


  def update(self):
    net = self.net
    (data, target) = next(self.iter)
    data, target = data.to(self.device), target.to(self.device)
    self.step_count+=1

    loss = net.loss(data, target)
    grads = autograd.grad(loss, net.W[0], create_graph=True)
    v = [torch.randn(w.shape).to(self.device) for w in net.W]
    Hv = autograd.grad(grads, net.W, v)

    psteps = []
    for j in range(len(net.W)):
      q = self.Qs[j]
      dw = v[j]
      dg = Hv[j]
      qleft, qright, pstep = psgd.update_precond_kron_with_step(q[0], q[1],
                                                                dw, dg)
      self.Qs[j][0] = qleft
      self.Qs[j][1] = qright
      psteps.append(pstep)

    print(self.step_count, np.array(psteps).mean())
    self.logger('p_residual', np.array(psteps).mean())


  def adjust_grads(self, grads):
    assert len(grads) == len(self.net.W)
    return [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(self.Qs, grads)]


def main():
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  device = torch.device("cuda" if use_cuda else "cpu")
  print("using device ", device)
  torch.manual_seed(args.seed)

  logger = u.TensorboardLogger(args.run)
  batch_size = 64
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/data', train=True, download=True,
                   transform=transforms.Compose([
                     transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True, **kwargs)
  preconditioner_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/data', train=True, download=True,
                   transform=transforms.Compose([
                     transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
      transforms.ToTensor()])),
    batch_size=1000, shuffle=True, **kwargs)

  """input image size for the original LeNet5 is 32x32, here is 28x28"""

  #  W1 = 0.1 * torch.randn(1 * 5 * 5 + 1, 6)

  net = LeNet5().to(device)
  preconditioner = Preconditioner(net, preconditioner_loader, device)

  def train_loss(data, target):
    y = net(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    for w in net.W:
      loss += 0.0002 * torch.sum(w * w)

    return loss

  def test_loss():
    num_errs = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        y = net(data)
        _, pred = torch.max(y, dim=1)
        num_errs += torch.sum(pred != target)

    return num_errs.item() / len(test_loader.dataset)

  Qs = [[torch.eye(w.shape[0]), torch.eye(w.shape[1])] for w in net.W]
  for i in range(len(Qs)):
    for j in range(len(Qs[i])):
      Qs[i][j] = Qs[i][j].to(device)

  step_size = 0.1  # tried 0.15, diverges
  grad_norm_clip_thr = 1e10
  TrainLoss, TestLoss = [], []
  example_count = 0
  step_time_ms = 0

  for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
      step_start = time.perf_counter()
      data, target = data.to(device), target.to(device)

      loss = train_loss(data, target)

      with u.timeit('grad'):
        grads = autograd.grad(loss, net.W, create_graph=True)
      TrainLoss.append(loss.item())
      logger.set_step(example_count)
      logger('loss/train', TrainLoss[-1])
      if batch_idx % 10 == 0:
        print(f'Epoch: {epoch}; batch: {batch_idx}; train loss: {TrainLoss[-1]:.2f}, step time: {step_time_ms:.0f}')

      with u.timeit('Hv'):
        #        noise.normal_()
        v = [torch.randn(w.shape).to(device) for w in net.W]
        # v = grads
        Hv = autograd.grad(grads, net.W, v)

      with u.timeit('P_update'):
        for i in range(num_updates):
          preconditioner.update()  # updates Q values
          pass


      with torch.no_grad():
        with u.timeit('g_update'):
          # preconditions gradient using current estimate of Qs
          pre_grads = preconditioner.adjust_grads(grads)
          grad_norm = torch.sqrt(sum([torch.sum(g * g) for g in pre_grads]))

        with u.timeit('gradstep'):
          step_adjust = min(grad_norm_clip_thr / (grad_norm + 1.2e-38), 1.0)
          for i in range(len(net.W)):
            net.W[i] -= step_adjust * step_size * pre_grads[i]

        total_step = step_adjust * step_size
        logger('step/adjust', step_adjust)
        logger('step/size', step_size)
        logger('step/total', total_step)
        logger('grad_norm', grad_norm)

      example_count += batch_size
      step_time_ms = 1000 * (time.perf_counter() - step_start)
      logger('time/step', step_time_ms)

      if args.test and batch_idx >= 100:
        break
    if args.test and batch_idx >= 100:
      break

    test_loss0 = test_loss()
    TestLoss.append(test_loss0)
    logger('loss/test', test_loss0)
    step_size = (0.1 ** 0.1) * step_size
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))

  if args.test:
    step_times = logger.d['time/step']
    assert step_times[-1] < 30, step_times  # should be around 20ms
    losses = logger.d['loss/train']
    assert losses[0] > 2  # around 2.3887393474578857
    assert losses[-1] < 0.5, losses
    print("Test passed")


if __name__ == '__main__':
  main()
