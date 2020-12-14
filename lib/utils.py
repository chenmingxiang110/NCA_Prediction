import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def tup_distance(node1, node2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5
    elif mode=="Manhattan":
        return np.abs(node1[0]-node2[0])+np.abs(node1[1]-node2[1])
    else:
        raise ValueError("Unrecognized distance mode: "+mode)

def mat_distance(mat1, mat2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return np.sum((mat1-mat2)**2, axis=-1)**0.5
    elif mode=="Manhattan":
        return np.sum(np.abs(mat1-mat2), axis=-1)
    else:
        raise ValueError("Unrecognized distance mode: "+mode)

def get_sobel(size):
    assert (size+1)%2==0
    gx = np.zeros([size, size])
    gy = np.zeros([size, size])
    mid = size//2
    
    for row in range(size):
        for col in range(size):
            i = col-mid
            j = row-mid
            gx[row, col] = i / max((i*i + j*j), 1e-6)
            gy[row, col] = j / max((i*i + j*j), 1e-6)
    
    return gx, gy

class SamplePool:

    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def get_living_mask(x, alpha_channel, kernel_size=3):
    return F.max_pool2d(x[:, alpha_channel:(alpha_channel+1), :, :],
                        kernel_size=kernel_size, stride=1, padding=(kernel_size//2)) > 0.1

def get_rand_avail(alive_map):
    a = np.where(alive_map>0)
    index = np.random.randint(len(a[0]))
    return (a[1][index], a[2][index])

def make_seed(shape, n_channels, alpha_channels, coord=None):
    if coord is None:
        coord = (shape[0]//2, shape[1]//2)
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[coord[0], coord[1], alpha_channels] = 1.0
    return seed

def make_circle_masks(n, h, w, rmin=0.2, rmax=0.4):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.random([2, n, 1, 1])*1.0-0.5
    r = np.random.random([n, 1, 1])*(rmax-rmin)+rmin
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = (x*x+y*y < 1.0).astype(np.float32)
    return mask

def softmax(x, axis):
    norm = np.exp(x-np.max(x, axis, keepdims=True))
    y = norm/np.sum(norm, axis, keepdims=True)
    return y
