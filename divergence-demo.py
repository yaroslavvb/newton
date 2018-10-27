"""
Epoch: 0; batch: 0; train loss: 2.3933441638946533
Epoch: 0; batch: 100; train loss: 0.2868724763393402
Epoch: 0; batch: 200; train loss: 0.19104759395122528
Epoch: 0; batch: 300; train loss: 0.3258479833602905
Epoch: 0; batch: 400; train loss: 0.19173741340637207
Epoch: 0; batch: 500; train loss: 0.15167993307113647
Epoch: 0; batch: 600; train loss: 0.20221514999866486
Epoch: 0; batch: 700; train loss: 0.12190920114517212
Epoch: 0; batch: 800; train loss: 0.14758999645709991
Epoch: 0; batch: 900; train loss: 0.17044204473495483
Epoch: 0; best test loss: 0.902
Elapsed time for this epoch: 18.70916986465454
Epoch: 1; batch: 0; train loss: nan
Epoch: 1; batch: 100; train loss: nan
"""

import torch
from torch.autograd import grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import preconditioned_stochastic_gradient_descent as psgd 
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
update_preconditioner_every = 1#update every # of iterations
update_preconditioner_times = 1#update # times

shuffle=False
torch.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,           
                       transform=transforms.Compose([                       
                               transforms.ToTensor()])),    
                        batch_size=64, shuffle=shuffle)
test_loader = torch.utils.data.DataLoader(    
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),    
                        batch_size=1000, shuffle=shuffle)

"""input image size for the original LeNet5 is 32x32, here is 28x28"""
W1 = torch.tensor(0.1*torch.randn(1*5*5+1,  6), requires_grad=True, device=device)
W2 = torch.tensor(0.1*torch.randn(6*5*5+1,  16), requires_grad=True, device=device)
W3 = torch.tensor(0.1*torch.randn(16*4*4+1, 120), requires_grad=True, device=device)#here is 4x4, not 5x5
W4 = torch.tensor(0.1*torch.randn(120+1,    84), requires_grad=True, device=device)
W5 = torch.tensor(0.1*torch.randn(84+1,     10), requires_grad=True, device=device)
Ws = [W1, W2, W3, W4, W5]

def LeNet5(x): 
    x = F.conv2d(x, W1[:-1].view(6,1,5,5), bias=W1[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.conv2d(x, W2[:-1].view(16,6,5,5), bias=W2[-1])
    x = F.relu(F.max_pool2d(x, 2))
    x = F.relu(x.view(-1, 16*4*4).mm(W3[:-1]) + W3[-1])
    x = F.relu(x.mm(W4[:-1]) + W4[-1])
    y = x.mm(W5[:-1]) + W5[-1]
    return y

def train_loss(data, target):
    y = LeNet5(data)
    y = F.log_softmax(y, dim=1)
    loss = F.nll_loss(y, target)
    for W in Ws:
        loss += 0.0002*torch.sum(W*W)
        
    return loss

def test_loss( ):
    num_errs = 0
    with torch.no_grad():
        for data, target in test_loader:
            y = LeNet5(data.to(device))
            _, pred = torch.max(y, dim=1)
            num_errs += torch.sum(pred!=target.to(device))
            
    return num_errs.item()/len(test_loader.dataset)

Qs = [[torch.eye(W.shape[0], device=device), torch.eye(W.shape[1], device=device)] for W in Ws]
step_size = 0.1
grad_norm_clip_thr = 1e8
TrainLoss, TestLoss = [], []
for epoch in range(10):
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_loss(data.to(device), target.to(device))
        TrainLoss.append(loss.item())
        if batch_idx%100==0:
            print('Epoch: {}; batch: {}; train loss: {}'.format(epoch, batch_idx, TrainLoss[-1]))
        
        grads = grad(loss, Ws, create_graph=True)
        if batch_idx%update_preconditioner_every == 0:
            for num_Qs_update in range(update_preconditioner_times):
                v = [torch.randn(W.shape, device=device) for W in Ws]
                Hv = grad(grads, Ws, grad_outputs=v, retain_graph=True)
                with torch.no_grad():
                    Qs = [psgd.update_precond_kron(q[0], q[1], dw, dg) for (q, dw, dg) in zip(Qs, v, Hv)]
        
        with torch.no_grad():           
            pre_grads = [psgd.precond_grad_kron(q[0], q[1], g) for (q, g) in zip(Qs, grads)]
            grad_norm = torch.sqrt(sum([torch.sum(g*g) for g in pre_grads]))
            step_adjust = min(grad_norm_clip_thr/(grad_norm + 1.2e-38), 1.0)
            for i in range(len(Ws)):
                Ws[i] -= step_adjust*step_size*pre_grads[i]
                
    TestLoss.append(test_loss())
    step_size = (0.5)*step_size
    print('Epoch: {}; best test loss: {}'.format(epoch, min(TestLoss)))
    print('Elapsed time for this epoch: {}'.format(time.time() - t0))
