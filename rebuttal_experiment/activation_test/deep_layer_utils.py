import collections
import time
import os
import pickle
import torch
import torch.nn as nn
import random
import numpy as np
import sys
import scipy.linalg
import functools
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEBUG = True
USE_GRADIENT = True

query_num = 0
LAYERS = 8
TINY = True
name_ID = sys.argv[2]

MODEL_NAME = "real_784-64x12-10"
FOLDER_NAME = "exp_deep12_pos2"
MODEL_ROOT_FOLDER_NAME = "models_list/"
EXP_ROOT_FOLDER_NAME = ""
RESULT_ROOT_FOLDER_NAME = ""

DEBUG_FILE = open("debug.txt","w")
RESULT_FILE = open(RESULT_ROOT_FOLDER_NAME +  "result" + str(name_ID) +".txt","w")

# I konw it's stupid and will be refined soon
if TINY:
    IDIM = 784
    # DIM Should be the max value of DIM1,DIM2,SHRINK 
    DIM = 64
    DIM1 = 64
    DIM2 = 64
    DIM3 = 64
    DIM4 = 64
    DIM5 = 64
    DIM6 = 64
    DIM7 = 64
    DIM8 = 64
    DIM9 = 64
    DIM10 = 64
    DIM11 = 64
    SHRINK = 64
else:
    IDIM = 32*32*3
    DIM = 256
    DIM1 = 30
    DIM2 = 30
    SHRINK = 64

SEED = 1 if len(sys.argv) < 3 else int(sys.argv[1])
print("Seed is " + str(SEED))

random.seed(SEED)
np.random.seed(SEED)

WEIGHTS_DIM_LIST = [IDIM,DIM1,DIM2,DIM3,DIM4,DIM5,DIM6,DIM7,DIM8,DIM9,DIM10,DIM11,SHRINK,10]

class MathIsHard(Exception):
    pass

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.output_DIM = [DIM1,DIM2,DIM3,DIM4,DIM5,DIM6,DIM7,DIM8,DIM9,DIM10,DIM11,SHRINK,10]
        self.fc1 = nn.Linear(IDIM, DIM1)
        self.fc2 = nn.Linear(DIM1, DIM2)
        self.fc3 = nn.Linear(DIM2, DIM3)
        self.fc4 = nn.Linear(DIM3, DIM4)
        self.fc5 = nn.Linear(DIM4, DIM5)
        self.fc6 = nn.Linear(DIM5, DIM6)
        self.fc7 = nn.Linear(DIM6, DIM7)
        self.fc8 = nn.Linear(DIM7, DIM8)
        self.fc9 = nn.Linear(DIM8, DIM9)
        self.fc10 = nn.Linear(DIM9, DIM10)
        self.fc11 = nn.Linear(DIM10, DIM11)
        self.fc12 = nn.Linear(DIM11, SHRINK)
        self.fc13 = nn.Linear(SHRINK, 10)
        # self.relu = nn.ReLU()
        self.prelu1 = nn.PReLU(num_parameters=DIM1)
        self.prelu2 = nn.PReLU(num_parameters=DIM2)
        self.prelu3 = nn.PReLU(num_parameters=DIM3)
        self.prelu4 = nn.PReLU(num_parameters=DIM4)
        self.prelu5 = nn.PReLU(num_parameters=DIM5)
        self.prelu6 = nn.PReLU(num_parameters=DIM6)
        self.prelu7 = nn.PReLU(num_parameters=DIM7)
        self.prelu8 = nn.PReLU(num_parameters=DIM8)
        self.prelu9 = nn.PReLU(num_parameters=DIM9)
        self.prelu10 = nn.PReLU(num_parameters=DIM10)
        self.prelu11 = nn.PReLU(num_parameters=DIM11)
        self.prelu12 = nn.PReLU(num_parameters=SHRINK)
        # self.prelu1.apply(init_prelu_param)
        # self.prelu2.apply(init_prelu_param)
        # self.prelu3.apply(init_prelu_param)
        # self.prelu4.apply(init_prelu_param)
        # self.prelu5.apply(init_prelu_param)
        # self.prelu6.apply(init_prelu_param)
        # self.prelu7.apply(init_prelu_param)
        # self.prelu8.apply(init_prelu_param)
        self.double()  # 使用双精度浮点数

    def first_layer(self,x):
        new_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if j < DIM1/2:
                    new_x[i][j] = max(x[i][j],0)
                else:
                    new_x[i][j] = min(0,x[i][j])
        return new_x

    @torch.no_grad
    def forward(self, x):
        prelu1 = self.prelu1
        prelu2 = self.prelu2
        prelu3 = self.prelu3
        prelu4 = self.prelu4
        prelu5 = self.prelu5
        prelu6 = self.prelu6
        prelu7 = self.prelu7
        prelu8 = self.prelu8
        prelu9 = self.prelu9
        prelu10 = self.prelu10
        prelu11 = self.prelu11
        prelu12 = self.prelu12
        x = x.view(-1, IDIM)
        x = prelu1(self.fc1(x))
        x = prelu2(self.fc2(x))
        x = prelu3(self.fc3(x))
        x = prelu4(self.fc4(x))
        x = prelu5(self.fc5(x))
        x = prelu6(self.fc6(x))
        x = prelu7(self.fc7(x))
        x = prelu8(self.fc8(x))
        x = prelu9(self.fc9(x))
        x = prelu10(self.fc10(x))
        x = prelu11(self.fc11(x))
        x = prelu12(self.fc12(x))
        x = self.fc13(x)
        return x

    def forward_grad(self, x):
        prelu1 = self.prelu1
        prelu2 = self.prelu2
        prelu3 = self.prelu3
        prelu4 = self.prelu4
        prelu5 = self.prelu5
        prelu6 = self.prelu6
        prelu7 = self.prelu7
        prelu8 = self.prelu8
        prelu9 = self.prelu9
        prelu10 = self.prelu10
        prelu11 = self.prelu11
        prelu12 = self.prelu12
        x = x.view(-1, IDIM)
        x = prelu1(self.fc1(x))
        x = prelu2(self.fc2(x))
        x = prelu3(self.fc3(x))
        x = prelu4(self.fc4(x))
        x = prelu5(self.fc5(x))
        x = prelu6(self.fc6(x))
        x = prelu7(self.fc7(x))
        x = prelu8(self.fc8(x))
        x = prelu9(self.fc9(x))
        x = prelu10(self.fc10(x))
        x = prelu11(self.fc11(x))
        x = prelu12(self.fc12(x))
        x = self.fc13(x)
        return x

    @torch.no_grad
    def cheat(self, x):
        o = []
        def prelu(x, layer):
            if x.size(-1) < DIM:
                padding = DIM - x.size(-1)
                xx = torch.nn.functional.pad(x, (0, padding))
                xx[:, -padding:] = 1
                o.append(xx)
            else:
                o.append(x)
            if layer == 1:
                return self.prelu1(x)
            elif layer == 2:
                return self.prelu2(x)
            elif layer == 3:
                return self.prelu3(x)
            elif layer == 4:
                return self.prelu4(x)
            elif layer == 5:
                return self.prelu5(x)
            elif layer == 6:
                return self.prelu6(x)
            elif layer == 7:
                return self.prelu7(x)
            elif layer == 8:
                return self.prelu8(x)
            elif layer == 9:
                return self.prelu9(x)
            elif layer == 10:
                return self.prelu10(x)
            elif layer == 11:
                return self.prelu11(x)
            elif layer == 12:
                return self.prelu12(x)

        x = x.view(-1, IDIM)
        x = prelu(self.fc1(x),1)
        x = prelu(self.fc2(x),2)
        x = prelu(self.fc3(x),3)
        x = prelu(self.fc4(x),4)
        x = prelu(self.fc5(x),5)
        x = prelu(self.fc6(x),6)
        x = prelu(self.fc7(x),7)
        x = prelu(self.fc8(x),8)
        x = prelu(self.fc9(x),9)
        x = prelu(self.fc10(x),10)
        x = prelu(self.fc11(x),11)
        x = prelu(self.fc12(x),12)
        return torch.stack(o)

    @torch.no_grad
    def cheat_full(self, x):
        o = []
        def prelu(x,layer):
            if x.size(-1) < DIM:
                padding = DIM - x.size(-1)
                xx = torch.nn.functional.pad(x, (0, padding))
                xx[:, -padding:] = 1
                o.append(xx)
            else:
                o.append(x)
            if layer == 1:
                o.append(self.prelu1(x))
                return self.prelu1(x)
            elif layer == 2:
                o.append(self.prelu2(x))
                return self.prelu2(x)
            elif layer == 3:
                o.append(self.prelu3(x))
                return self.prelu3(x)
            elif layer == 4:
                o.append(self.prelu4(x))
                return self.prelu4(x)
            elif layer == 5:
                o.append(self.prelu5(x))
                return self.prelu5(x)
            elif layer == 6:
                o.append(self.prelu6(x))
                return self.prelu6(x)
            elif layer == 7:
                o.append(self.prelu7(x))
                return self.prelu7(x)
            elif layer == 8:
                o.append(self.prelu8(x))
                return self.prelu8(x)
            elif layer == 9:
                o.append(self.prelu9(x))
                return self.prelu9(x)
            elif layer == 10:
                o.append(self.prelu10(x))
                return self.prelu10(x)
            elif layer == 11:
                o.append(self.prelu11(x))
                return self.prelu11(x)
            elif layer == 12:
                o.append(self.prelu12(x))
                return self.prelu12(x)

        x = x.view(-1, IDIM)
        x = prelu(self.fc1(x),1)
        x = prelu(self.fc2(x),2)
        x = prelu(self.fc3(x),3)
        x = prelu(self.fc4(x),4)
        x = prelu(self.fc5(x),5)
        x = prelu(self.fc6(x),6)
        x = prelu(self.fc7(x),7)
        x = prelu(self.fc8(x),8)
        x = prelu(self.fc9(x),9)
        x = prelu(self.fc10(x),10)
        x = prelu(self.fc11(x),11)
        x = prelu(self.fc12(x),12)
        return torch.stack(o)



cheat_net_cuda = CIFAR10Net()
if not TINY:
    cheat_net_cuda.load_state_dict(torch.load("new_models/final.pth"))
else:
    cheat_net_cuda.load_state_dict(torch.load(MODEL_ROOT_FOLDER_NAME + MODEL_NAME + ".pth"))
cheat_net_cuda.double().cuda()

cheat_net_cpu = CIFAR10Net()
if not TINY:
    cheat_net_cpu.load_state_dict(torch.load("new_models/final.pth"))
else:
    cheat_net_cpu.load_state_dict(torch.load(MODEL_ROOT_FOLDER_NAME + MODEL_NAME + ".pth"))
cheat_net_cpu.double()

cheat_solution = [x.cpu().detach().numpy() for x in cheat_net_cpu.parameters()][::2]

pytorch_state = cheat_net_cpu.state_dict()
WEIGHTS = []
BIASES = []
SLOPES = []
for i in range(1,9):
    weight_index = f'fc{i}.weight'
    bias_index = f'fc{i}.bias'
    pytorch_weight = pytorch_state[weight_index].cpu().numpy()
    pytorch_bias = pytorch_state[bias_index].cpu().numpy()

    keras_weight = pytorch_weight.T
    keras_bias = pytorch_bias
    WEIGHTS.append(keras_weight)
    BIASES.append(keras_bias)
    if i != 9:
        slope_index = f'prelu{i}.weight'
        pytorch_slope = pytorch_state[slope_index].cpu().numpy()
        SLOPES.append(pytorch_slope)

w_last = pytorch_state['fc9.weight'].cpu().numpy()
w_last = w_last.T
WEIGHTS.append(w_last)

b_last = pytorch_state['fc9.bias'].cpu().numpy()
BIASES.append(b_last)
print("generating value ok!",file=DEBUG_FILE)


def model(x):
    global query_num
    assert len(x.shape) == 1
    query_num = query_num + 1
    return (cheat_net_cpu(torch.tensor(x).double()).numpy().argmax(1)).astype(np.int32)[0]
    
def bmodel(x):
    global query_num
    query_num = query_num + len(x)
    return (cheat_net_cuda(x).argmax(1)).to(torch.int32)

def cheat(x):
    if not DEBUG: raise
    return cheat_net_cpu.cheat(torch.tensor(x).double()).numpy()

def cheat_cuda(x):
    if not DEBUG: raise
    return cheat_net_cuda.cheat(x)

def gap(x):
    if not DEBUG: raise
    out = cheat_net_cpu(torch.tensor(x).double()).numpy()
    #print(out)
    max_idx = np.argmax(out, 1)
    top = out[np.arange(len(out)), max_idx]
    out[np.arange(len(out)), max_idx] = -100
    return top - np.max(out, 1)

def gapt(x, grad=False):
    if not DEBUG: raise
    if grad:
        out = cheat_net_cuda.forward_grad(x)
    else:
        out = cheat_net_cuda(x)

    max_idx = torch.argmax(out, 1)
    top = out[torch.arange(len(out)), max_idx]
    out[torch.arange(len(out)), max_idx] = -100
    return top - torch.max(out, 1).values

def cheat_num_flips(a,b):
    if not DEBUG: raise
    return np.sum((cheat(a)>0) != (cheat(b)>0))

def cheat_neuron_diff(a,b):
    if not DEBUG: raise
    return np.where((cheat(a)>0).flatten() != (cheat(b)>0).flatten())[0]

def cheat_neuron_diff_cuda(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cuda()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy()

def cheat_neuron_diff_cuda_2(a,b):
    if not DEBUG: raise
    ab = torch.tensor(np.stack([a,b])).double().cuda()
    out = cheat_cuda(ab)>0
    a = out[:,0,:]
    b = out[:,1,:]
    return torch.where(a.flatten() != b.flatten())[0].cpu().numpy(), np.array(a.flatten().cpu().numpy(), dtype=np.uint8)


# Find the decision boundary, given two inputs:
#    zero must have model(zero) = 0
#    one  must have model(one)  = 1
# returns the midpoint that's almost 0/1 at the same time
def find_decision_boundary(zero=None, one=None, tensor=False):
    if zero is None and one is None:
        points = {}
        while len(points) < 2:
            if not TINY:
                pass
                # maybe = random.sample(range(len(x_test)), 10)
                # maybe = x_test[maybe]
            else:
                maybe = np.random.normal(size=(10, IDIM))
            maybe = torch.tensor(maybe).cuda().double()
            outs = bmodel(maybe)
            for out, point in zip(outs, maybe):
                points[out.item()] = point
        zero, one = list(points.values())[:2]
    #assert model(zero) != model(one)

    model_zero = bmodel(zero)
    last = 1e9
    while torch.sum(torch.abs(zero - one)) > 1e-16 and torch.sum(torch.abs(zero - one)) < last:
        last = torch.sum(torch.abs(zero - one))
        mid = (zero+one)/2
        if bmodel(mid) == model_zero:
            zero = mid
        else:
            one = mid


    if tensor:
        return zero
    return zero.cpu().numpy()

def find_decision_boundary_batched(zero, one):
    zero = torch.tensor(zero).double().cuda()
    one = torch.tensor(one).double().cuda()
    last = torch.tensor(1e9).cuda()

    orig_label = bmodel(zero)[0]
    
    while True:
        s = torch.sum(torch.abs(zero - one), dim=1)
        if not torch.any((s > 1e-14) | (s < last)).item():
            break
        
        last = s
        mid = (zero + one) / 2
        
        idx = bmodel(mid)
        
        zero_mask = (idx == orig_label)
        one_mask = (idx != orig_label)
        
        zero[zero_mask] = mid[zero_mask]
        one[one_mask] = mid[one_mask]

    return zero

# Compute the gradient direction of the decision boundary
# More correctly, this function returns a parallel direction
# to the hyperplane, so we can walk along it.
def get_gradient_dir(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    original = model(x)

    if dimensions is None:
        dimensions = range(IDIM)
    
    ratios = []
    for i in dimensions:
        if debug:
            print('iter', i)
        xp = np.array(x)


        if debug:
            print('start', gap(xp))
            sig = np.sign(cheat(x).flatten())

        xp[0] += step_size
        if model(xp) != original:
            xp[0] -= step_size*2

        if debug:
            print('   ', gap(xp))
            print('sigchange', np.sum(sig!= np.sign(cheat(xp).flatten())))

        #assert model(xp) == 0

        for step in 10**np.arange(-7, 0, .33):
            xp2 = np.array(xp)
            xp2[i] += step
            if model(xp2) == original:
                xp2[i] -= 2*step
            if model(xp2) != original:
                break
        else:
            assert False

        if debug:
            print('   ', gap(xp2))

        boundary = find_decision_boundary(xp, xp2)

        ratio = (xp[i]-boundary[i])/(step_size)
        ratios.append(ratio)

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def vectorized_right_boundary_search(left, orig_label, dimensions):
    device = 'cuda'
    batch_size = IDIM
    
    # Initialize the batch with copies of the left boundary
    batch = left.repeat(batch_size, 1)
    
    # Create a mask to keep track of dimensions that haven't found the boundary yet
    active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    active_mask[dimensions] = 1
    #print(active_mask)
    
    for step in 10**np.arange(-7, 0, .33):
        #print(step)
        # Try positive step
        pos_batch = batch.clone()
        pos_batch[active_mask] += torch.eye(IDIM, device=device)[active_mask] * step

        pos_results = bmodel(pos_batch)
        #print('aa',pos_results)
        
        # Update batch and mask for positive steps that crossed the boundary
        pos_crossed = (pos_results != orig_label) & active_mask
        #print(pos_crossed)
        #print('act',active_mask)
        #print("Set", torch.where(pos_crossed), torch.sum(pos_crossed.float()))
        batch[pos_crossed] = pos_batch[pos_crossed]
        active_mask[pos_crossed] = False
        
        if not active_mask.any():
            break
        
        # Try negative step for remaining active dimensions
        neg_batch = batch.clone()
        neg_batch[active_mask] -= torch.eye(IDIM, device=device)[active_mask] * step
        
        neg_results = bmodel(neg_batch)
        #print('bb',neg_results)
        
        # Update batch and mask for negative steps that crossed the boundary
        neg_crossed = (neg_results != orig_label) & active_mask
        #print(neg_crossed)
        #print('act',active_mask)
        #print("Set", torch.where(neg_crossed), torch.sum(neg_crossed.float()))

        batch[neg_crossed] = neg_batch[neg_crossed]
        active_mask[neg_crossed] = False
        
        if not active_mask.any():
            break
    
    if active_mask.any():
        raise MathIsHard("Boundary not found for all dimensions")
    
    # Convert results back to numpy arrays
    #print(batch.shape)
    #print('zz',left - batch[0])
    return left.repeat(IDIM, 1)[dimensions, :], batch[dimensions, :]

def get_gradient_dir_fast(x, cache={}, debug=False, step_size=1e-7, dimensions=None):
    if len(cache) > 1e6:
        cache.clear()
    if tuple(x) in cache:
        return cache[tuple(x)]

    if dimensions is None:
        dimensions = range(IDIM)
    
    #np.save("/tmp/x.npy", x)
    
    # 1. init
    # find the left and right sides
    leftright = []

    left = np.array(x)
    left = torch.tensor(left, dtype=torch.float64, device='cuda')
    original = bmodel(left)

    left[0] += step_size
    if bmodel(left) != original:
        left[0] -= step_size*2
        
    #assert model(left) == 0
    
    xp, xp2 = vectorized_right_boundary_search(left, original, dimensions)

    ratios = []
    #xp, xp2 = zip(*leftright)


    boundary = find_decision_boundary_batched(xp, xp2)

    ratios = ((xp-boundary)/(step_size))[torch.arange(len(dimensions)), dimensions].cpu().numpy()

    # So far we have the gradient direction, now let's make is
    # so that we can go parallel
    ratios = np.array(ratios)

    cache[tuple(x)] = ratios

    return ratios

def get_normal(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cuda(), grad=True)
        out[0].backward()
        real = np.random.normal(0, 1) * x.grad.cpu().numpy()
        real = norm(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)
    
        return fnormal

def get_normal_t(x, step_size=1e-6):
    if USE_GRADIENT:
        x = torch.tensor(x, requires_grad=True)
        out = gapt(x.cuda(), grad=True)
        out[0].backward()
        real = torch.tensor(np.random.normal(0, 1)) * x.grad
        real = normt(real)
        return real
    else:
        try:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size)
        except MathIsHard:
            fnormal = 1/get_gradient_dir_fast(x, step_size=step_size/10)
        fnormal = norm(fnormal)
    
        return fnormal
    
def norm(x):
    return x / np.sum(x**2)**.5

def normt(x):
    return x / torch.sum(x**2)**.5
