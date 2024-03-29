import numpy as np
import functools
import mytorch.autograd_engine as autograd_engine
from mytorch.nn import functional as F
from mytorch.nn.comp_graph import ForwardGraphVisualizer, BackwardGraphVisualizer


def cat(seq,dim=0):
    '''
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    Args:
        seq (list of Tensors) - List of interegers to concatenate
        dim (int) - The dimension along which to concatenate
    Returns:
        Tensor - Concatenated tensor

    Example:

        seq
        [[[3 3 4 1]
        [0 3 1 4]],
        [[4 2 0 0]
        [3 0 4 0]
        [1 4 4 3]],
        [[3 2 3 1]]]
        
        tensor.cat(seq,0)
        [[3 3 4 1]
        [0 3 1 4]
        [4 2 0 0]
        [3 0 4 0]
        [1 4 4 3]
        [3 2 3 1]]
    '''
    return F.Cat.apply(seq, dim)

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
                 is_parameter=False, name = None, op = None):
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
        self.name, self.op = name, op
        self.children = []

    @functools.lru_cache()
    def build_graph_topology(self):
        def dfs(node, visited, nodes):
            visited.add(node)
            if node.ctx:
                for parent in node.ctx.parents:
                    if parent not in visited:
                        dfs(parent, visited, nodes)
                nodes.append(node)
            return nodes
        return dfs(self, set(), list())
    
    def bkwrd(self):
        if self.shape != (1,):
            raise Exception(
                "Can't initiate backprop from a non scalar-valued tensor.")

        self.grad = Tensor.ones(self.shape, requires_grad=False)

        for node in reversed(self.build_graph_topology()):
            assert node.grad is not None, 'Got an unitialized gradient node'

            gradients = node.ctx.backward(node.ctx, node.grad.data)

            if len(node.ctx.parents) == 1:
                gradients = [gradients]

            for tensor, grad in zip(node.ctx.parents, gradients):
                if grad is not None:
                    assert grad.shape == tensor.shape, f"Mismatched tensor and grad shape. Got {grad.shape} and {tensor.shape}. \
                                                         Tensor and gradient should have the same shape."
                    if tensor.grad is None:
                        tensor.grad = Tensor(grad, requires_grad=False)
                    else:
                        tensor.grad += Tensor(grad, requires_grad=False)
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

    
    def __getitem__(self, key):
        return F.Slice.apply(self, key)


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

    def split(self, batches):
        """Splits a tensor into batches and returns a list of tensors"""
        ls = []
        i_p = 0
        indices = np.cumsum(batches).astype(int)
        for i in range(len(indices)):
            ls.append(self[i_p:indices[i]])
            i_p = indices[i]
        return ls

    def flatten(self):
        return F.flatten(self)



    # Below methods can be used WITHOUT creating a tensor first
    # (For example, we can call Tensor.zeros(3,2) directly)

    @classmethod
    def zeros(cls,*shape, **kwargs):
        """Creates new tensor filled with 0's
        Args:
            shape: comma separated ints i.e. Tensor.zeros(3,4,5)
        Returns:
            Tensor: filled w/ 0's
        """
        return cls(np.zeros(shape), **kwargs)

    @classmethod
    def ones(cls,*shape, **kwargs):
        """Creates new tensor filled with 1's
        Note: if you look up "asterik args python", you'll see this function is
        called as follows: ones(1, 2, 3), not: ones((1, 2, 3))
        """
        return cls(np.ones(shape), **kwargs)

    @staticmethod
    def arange(*interval):
        """Creates new tensor filled by `np.arange()`"""
        return Tensor(np.arange(*interval))

    @classmethod
    def randn(cls,*shape, **kwargs):
        """Creates new tensor filled by normal distribution (mu=0, sigma=1)"""
        return cls(np.random.normal(0, 1, shape), **kwargs)

    @staticmethod
    def empty(*shape):
        """Creates an tensor with uninitialized data (NOT with 0's).

        >>> Tensor.empty(1,)
        [6.95058141e-310]
        """
        return Tensor(np.empty(shape))
    
    @classmethod
    def normal(cls, loc, scale, *shape, **kwargs):
        return cls(np.random.normal(loc, scale, *shape), **kwargs)
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

    def __len__(self,):
        return len(self.data)
  

    def exp(self):
        """Element-wise exp of this tensor, adding to comp graph"""
        return F.Exp.apply(self)
    # def acc(self):
    #     return F.AccumulateGrad(self)
    
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

    def unsqueeze(self,dim=0):
        """ 
        Returns a new tensor with a dimension of size one inserted at the specified position. 
        
        NOTE: If you are not sure what this operation does, please revisit Recitation 0.
        
        Example:
            a
            [[1 2 3]
            [4 5 6]]
            
            a.unsqueeze(0)
            [[[1 2 3]
            [4 5 6]]]
            
            a.unsqueeze(1)
            [[[1 2 3]]
            
            [[4 5 6]]]
            
            a.unsqueeze(2)
            [[[1]
            [2]
            [3]]
            
            [[4]
            [5]
            [6]]]
        """
        shape = self.shape[:dim] + (1,) + self.shape[dim:] # insert 1 into shape at dim
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






    # ****************************************
    # ******** Activation functions **********
    # ****************************************

    def relu(self):
        return F.ReLU.apply(self)
    
    def sigmoid(self):
        return F.Sigmoid.apply(self)

    def tanh(self):
        return F.Tanh.apply(self)

    # ****************************************
    # ************ Visualization *************
    # ****************************************

    def plot_forward(self, rankdir="TB"):
        r"""
            Plots a forward computational graph

            Args:
                rankdir (str): LR (left to right) and TB (top to bottom)
        """
        visualizer = ForwardGraphVisualizer()
        return visualizer.visualize(self, rankdir=rankdir)
    
    def plot_backward(self, rankdir="LR"):
        r"""
            Plots a backward computational graph

            Args:
                rankdir (str): LR (left to right) and TB (top to bottom)
        """
        visualizer = BackwardGraphVisualizer()
        return visualizer.visualize(self, rankdir=rankdir)
