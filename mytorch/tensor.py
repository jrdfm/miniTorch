import numpy as np

import mytorch.autograd_engine as autograd_engine
from mytorch.nn import functional as F



class Tensor:
    """Tensor object, similar to `torch.Tensor`
    A wrapper around a np array that help it interact with MyTorch.

    Args:
        data (np.array): the actual data of the tensor
        requires_grad (boolean): If true, accumulate gradient in `.grad`
        is_leaf (boolean): If true, this is a leaf tensor; see writeup.
        is_parameter (boolean): If true, data contains trainable params
    """
    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False):
        if  type(data).__name__ == 'ndarray':
            self.data = data
        else:
            self.data = np.array(data)
            
        # print(type(data))
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    # ------------------------------------
    # [Not important] For printing tensors
    # ------------------------------------
    def __str__(self):
        return "{}{}".format(
            str(self.data),
            ", grad_fn={}".format(self.grad_fn.__class__.__name__) if self.grad_fn is not None else ""
        )

    def __repr__(self):
        return self.__str__()

    
    def __getitem__(self, args):
        if args is None: 
            args = []
        elif type(args) in [list, tuple]: 
            pass
        else: 
            args = [args]

        indices = []

        for i, arg in enumerate(args):
            start, stop = arg.start, arg.stop

            if start is None:
                start = 0
            elif not np.issubdtype(type(start), int):
                raise TypeError(f"Indices must be integer. Got {type(start)}")
            
            if stop is None:
                stop = self.shape[i]
            elif not np.issubdtype(type(stop), int):
                raise TypeError(f"Indices must be integer. Got {type(stop)}")
            elif stop < 0:
                stop = self.shape[i] + stop

            assert arg.step is None or arg.step == 1, "Custom step not yet implemented"
            indices.append((start, stop))
        
        indices += [(0, self.shape[i]) for i in range(len(args), len(self.shape))]
        
        return F.Slice.apply(self, indices)


    # ------------------------------------------
    # Tensor Operations (NOT part of comp graph)
    # ------------------------------------------
    @property
    def shape(self):
        """Returns the shape of the data array in a tuple.
        >>> a = Tensor(np.array([3,2])).shape
        (2,)
        """
        return self.data.shape
    @property
    def strides(self):
        """Returns Tuple of bytes to step in each dimension when traversing an array
        """
        return self.data.strides

    def fill_(self, fill_value):
        """In-place operation, replaces data with repeated value"""
        self.data.fill(fill_value)
        return self

    def copy(self):
        """Returns copy of this tensor
        Note: after copying, you may need to set params like `is_parameter` manually"""
        return Tensor(self.data)

    # Below methods can be used WITHOUT creating a tensor first
    # (For example, we can call Tensor.zeros(3,2) directly)

    @staticmethod
    def zeros(*shape):
        """Creates new tensor filled with 0's
        Args:
            shape: comma separated ints i.e. Tensor.zeros(3,4,5)
        Returns:
            Tensor: filled w/ 0's
        """
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape):
        """Creates new tensor filled with 1's
        Note: if you look up "asterik args python", you'll see this function is
        called as follows: ones(1, 2, 3), not: ones((1, 2, 3))
        """
        return Tensor(np.ones(shape))

    @staticmethod
    def arange(*interval):
        """Creates new tensor filled by `np.arange()`"""
        return Tensor(np.arange(*interval))

    @staticmethod
    def randn(*shape):
        """Creates new tensor filled by normal distribution (mu=0, sigma=1)"""
        return Tensor(np.random.normal(0, 1, shape))

    @staticmethod
    def empty(*shape):
        """Creates an tensor with uninitialized data (NOT with 0's).

        >>> Tensor.empty(1,)
        [6.95058141e-310]
        """
        return Tensor(np.empty(shape))

    # ----------------------
    # Autograd backward init
    # ----------------------
    def backward(self):
        """Kicks off autograd backward (see writeup for hints)"""
        #raise Exception("TODO: Kick off `autograd_engine.backward()``")
        autograd_engine.backward(self.grad_fn, Tensor(np.ones(self.shape)))

    # ------------------------------------------
    # Tensor Operations (ARE part of comp graph)
    # ------------------------------------------
    def T(self):
        """Transposes data (for 2d data ONLY)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).T()
        [[1, 4],
         [2, 5],
         [3, 6]]
        """
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        """Makes new tensor of input shape, containing same data
        (NOT in-place operation)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).reshape(3,2)
        [[1, 2],
         [3, 4],
         [5, 6]]
        """
        return F.Reshape.apply(self, shape)

    def log(self):
        """Element-wise log of this tensor, adding to comp graph"""
        return F.Log.apply(self)

    def __add__(self, other):
        """Links "+" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after adding
        """
        return F.Add.apply(self, other)

    def __sub__(self, other):
        #raise Exception("TODO: Link '-' to comp. graph, like in __add__()")
        return F.Sub.apply(self,other)
    
    def __mul__(self, other):
    
        return F.Mul.apply(self,other)

    def __truediv__(self, other):
        return F.Div.apply(self, other)    

    def __pow__(self, other):
        return F.Pow.apply(self,other)

    def __matmul__(self, other):
        return F.MatMul.apply(self, other)    

    def exp(self):
        """Element-wise exp of this tensor, adding to comp graph"""
        return F.Exp.apply(self)
    def acc(self):
        return F.AccumulateGrad(self)
    
    def sqrt(self):
        return F.Sqrt.apply(self)    

    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)
    
    def max(self, axis=None):
        return F.Max.apply(self, axis)
    
    def min(self, axis=None):
        return F.Min.apply(self, axis)
    def mean(self, axis=None, keepdims:bool=False):
        out = self.sum(axis=axis)
        coeff = np.prod(out.shape) / np.prod(self.shape)
        return out * Tensor(coeff)
        
    def reshape(self, shape:tuple):
        return F.Reshape.apply(self, shape)

    
    # ****************************************
    # ********* Conv/Pool operations *********
    # ****************************************


    
    def max_pool2d(self, kernel_size, stride = None):
        """MaxPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return F.MaxPool2d.apply(self, kernel_size, stride)
    
    def avg_pool2d(self, kernel_size:tuple=(2, 2), stride = None):
        """AvgPooling2d operation
        
            Args:
                kernel_size (tuple): Kernel length for pooling operation
        """
        return F.AvgPool2d.apply(self, kernel_size, stride)

    def pad1d(self, pad:tuple):
        """Padding for one-dimensional signal

        Args:
            pad (tuple): Amount of padding before and after the signal

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[0]:int(self.shape[2])+pad[1]]

    def pad2d(self, pad:tuple):
        """Padding for two-dimensional images

        Args:
            pad (tuple): 4-dimensional tuple. Amount of padding to be applied before and
                         after the signal along 2 dimensions

        Returns:
            Tensor: Padded signal
        """
        return self[:, :, -pad[2]:int(self.shape[2])+pad[3], -pad[0]:int(self.shape[3])+pad[1]]

    def conv1d(self, weight, stride:int):
        """1d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, kernel_length)
            stride (int): Stride of the convolution operation
        """
        return F.Conv1d.apply(self, weight, stride)
    
    def conv2d(self, weight, stride):
        """2d convolution
        
        Args:
            weight (Tensor): Filter weight (out_channel, in_channel, *kernel_size)
            stride (int): Stride of the convolution operation
        """
        return F.Conv2d.apply(self, weight, stride)





