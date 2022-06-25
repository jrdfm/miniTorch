#!/usr/local/bin/python

from mytorch.tensor import Tensor
from mytorch.nn.module import BatchNorm2d

import numpy as np
import torch

# BatchNorm2d test

if __name__ == "__main__":

    input = torch.randn(20, 100, 35, 45)
    i = Tensor(input.numpy())

    m_af = torch.nn.BatchNorm2d(100)
    m = torch.nn.BatchNorm2d(100, affine=False)
    m_mi = BatchNorm2d(100)

    out_af = m_af(input)
    out = m(input)
    out_mi = m_mi(i)

    print(f'out_af == out_mi {np.array_equal(out_af.detach().numpy(), out_mi.data)}  close {np.allclose(out_af.detach().numpy(), out_mi.data)}')
    print(f'out == out_mi {np.array_equal(out.numpy(), out_mi.data)}  close {np.allclose(out.numpy(), out_mi.data)}')