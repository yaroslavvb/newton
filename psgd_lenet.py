import time

import torch
from torch.autograd import grad
import torch.nn.functional as F
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
args = parser.parse_args()


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
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
      transforms.ToTensor()])),
    batch_size=1000, shuffle=True, **kwargs)

  """input image size for the original LeNet5 is 32x32, here is 28x28"""

  #  W1 = 0.1 * torch.randn(1 * 5 * 5 + 1, 6)
  W1 = torch.tensor(0.1 * torch.randn(1 * 5 * 5 + 1, 6), requires_grad=True)
  W2 = torch.tensor(0.1 * torch.randn(6 * 5 * 5 + 1, 16), requires_grad=True)
  W3 = torch.tensor(0.1 * torch.randn(16 * 4 * 4 + 1, 120), requires_grad=True)  # so here is 4x4, not 5x5
  W4 = torch.tensor(0.1 * torch.randn(120 + 1, 84), requires_grad=True)
  W5 = torch.tensor(0.1 * torch.randn(84 + 1, 10), requires_grad=True)
  Ws = [W1, W2, W3, W4, W5]
  W=[None, W1, W2, W3, W4, W5]

  #  for i in range(1, len(Ws)):
  #    Ws[i] = Ws[i].to(device)

  def LeNet5(x):
    x = F.conv2d(x, W1[:-1].view(6, 1, 5, 5), bias=W1[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.conv2d(x, W2[:-1].view(16, 6, 5, 5), bias=W2[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.relu(x.view(-1, 16 * 4 * 4).mm(W3[:-1]) + W3[-1])
    x = F.relu(x.mm(W4[:-1]) + W4[-1])
    y = x.mm(W5[:-1]) + W5[-1]
    return y

  def train_loss(data, target):
    y = LeNet5(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    for W in Ws:
      loss += 0.0002 * torch.sum(W * W)

    return loss

  def test_loss():
    num_errs = 0
    with torch.no_grad():
      for data, target in test_loader:
        y = LeNet5(data)
        _, pred = torch.max(y, dim=1)
        num_errs += torch.sum(pred != target)

    return num_errs.item() / len(test_loader.dataset)

  Qs = [[torch.eye(W.shape[0]), torch.eye(W.shape[1])] for W in Ws]
  step_size = 0.1  # tried 0.15, diverges
  grad_norm_clip_thr = 1e10
  TrainLoss, TestLoss = [], []
  example_count = 0
  step_time_ms = 0
  
  for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
      step_start = time.perf_counter()
      #      data, target = data.to(device), target.to(device)
      
      loss = train_loss(data, target)

      grads = grad(loss, Ws, create_graph=True)
      TrainLoss.append(loss.item())
      logger.set_step(example_count)
      logger('loss/train', TrainLoss[-1])
      if batch_idx % 10 == 0:
        print(f'Epoch: {epoch}; batch: {batch_idx}; train loss: {TrainLoss[-1]:.2f}, step time: {step_time_ms:.0f}')

      v = [torch.randn(W.shape) for W in Ws]
      Hv = grad(grads, Ws, v)  # let Hv=grads if using whitened gradients
      with torch.no_grad():
        Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
        grad_norm = torch.sqrt(sum([torch.sum(g * g) for g in pre_grads]))

        step_adjust = min(grad_norm_clip_thr / (grad_norm + 1.2e-38), 1.0)
        for i in range(len(Ws)):
          Ws[i] -= step_adjust * step_size * pre_grads[i]

        total_step = step_adjust * step_size
        logger('step/adjust', step_adjust)
        logger('step/size', step_size)
        logger('step/total', total_step)
        logger('grad_norm', grad_norm)

      example_count += batch_size
      step_time_ms = 1000*(time.perf_counter() - step_start)
      logger('time/step', step_time_ms)

    test_loss0 = test_loss()
    TestLoss.append(test_loss0)
    logger('loss/test', test_loss0)
    step_size = (0.1 ** 0.1) * step_size
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))


if __name__ == '__main__':
  main()
        
