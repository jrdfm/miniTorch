from turtle import shape
import numpy as np
import functools as f
from mytorch import tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad:np.ndarray, shape:tuple, to_keep:int=0) -> np.ndarray:
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensorize(a.data.T, requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, grad_output):
        return tensorize(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensorize(a.data.reshape(shape), requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, grad_output):
        return tensorize(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensorize(np.log(a.data), requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return tensorize(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        out = tensorize(a.data + b.data, requires_grad, is_leaf)
        out.children, out.op  = [a, b], 'add'
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data
        grad_a, grad_b = map(tensorize ,map(unbroadcast,[grad_a, grad_b], [a.shape, b.shape]))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors & of equal shape: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a ,b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return  tensorize(a.data - b.data,requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data
        grad_a, grad_b = map(tensorize ,map(unbroadcast,[grad_a, grad_b], [a.shape, b.shape]))
        return grad_a, grad_b

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        out = tensorize(a.data * b.data, requires_grad, is_leaf)
        out.children, out.op = [a, b], 'mul'
        return out 

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data
        grad_a, grad_b = map(tensorize ,map(unbroadcast,[grad_a, grad_b], [a.shape, b.shape]))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims = None):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only Sum of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        out = a.data.sum(axis = axis, keepdims = keepdims)
        return tensorize(out, requires_grad, is_leaf)

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data
        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()
        grad = np.ones(ctx.shape) * grad_out
        assert grad.shape == ctx.shape
        # gradient tensors SHOULD NEVER have requires_grad = True.
        return tensorize(grad), None, None

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf=not requires_grad
        c = tensorize(a.data / b.data, requires_grad, is_leaf)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = grad_output.data / b.data
        # dL/db = dout/db * dL/dout
        grad_b = (-a.data * grad_output.data) / (b.data ** 2)
        # the order of gradients returned should xch the order of the arguments
        grad_a, grad_b = map(tensorize ,map(unbroadcast,[grad_a, grad_b], [a.shape, b.shape]))
        return grad_a, grad_b

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("a must be Tensors: {}".format(type(a).__name__))
        if (type(b).__name__ == 'int' or type(b).__name__ == 'float'):
            b = tensor.Tensor(np.array(b))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf=not requires_grad
        return tensorize(a.data**b.data, requires_grad, is_leaf)
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a =  b.data * (a.data ** (b.data - 1)) * grad_output.data
        grad_a = tensorize(unbroadcast(grad_a, a.shape))
        return grad_a, None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that arg is a tensor
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("arg must be a Tensor: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad 
        is_leaf=not requires_grad
        return tensorize(np.exp(a.data), requires_grad, is_leaf)
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_a =  np.exp(a.data) * grad_output.data
        return tensorize(grad_a), None

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
                    a.data.shape[1] != b.data.shape[0]:
                    raise Exception("Both args must be Tensors & of same shape: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a ,b)
        requires_grad = a.requires_grad or b.requires_grad
        is_leaf = not requires_grad
        return tensorize(a.data @ b.data, requires_grad, is_leaf )
    @staticmethod
    def backward(ctx, grad_output):
        a, b  = ctx.saved_tensors
        grad_a = tensorize(grad_output.data @ b.data.T)
        grad_b = tensorize(a.data.T @ grad_output.data)
        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that arg is a tensor
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("arg must be a Tensor: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad 
        is_leaf= not requires_grad
        out = tensorize(np.where(a.data > 0,a.data, 0),requires_grad, is_leaf)
        out.children, out.op = [a], 'Relu' 
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a,  = ctx.saved_tensors
        grad_a = tensorize(np.where(a.data > 0, 1, 0)* grad_output.data)
        return grad_a

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensorize(b_data, a.requires_grad)
        b.is_leaf = not b.requires_grad
        b.children, b.op = [a], 'Sigmoid'
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensorize(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensorize(np.tanh(a.data), a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensorize(grad)
       
class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Sqrt must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        is_leaf = not requires_grad
        return tensorize(np.sqrt(a.data),requires_grad,is_leaf)
            
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        output = tensorize(1/2 *(a.data**(-1/2)) * grad_output.data)
        return output

class Max(Function):
    @staticmethod
    def forward(ctx, a, axis = None):
        requires_grad, is_leaf  = a.requires_grad, not a.requires_grad
        axis = [axis] if type(axis) == int else axis
        ctx.save_for_backward(a)
        out = max_min_forward(a.data, axis, 'max')
        out = tensorize(out, requires_grad, is_leaf)
        ctx.axis, ctx.out = axis, out.data
        return out

    @staticmethod
    def backward(ctx, grad_output):
        axis, out = ctx.axis, ctx.out
        a, = ctx.saved_tensors
        return max_min_backward(grad_output, a.data, out, axis)
        

class Min(Function):
    @staticmethod
    def forward(ctx, a, axis = None):
        requires_grad, is_leaf  = a.requires_grad, not a.requires_grad
        axis = [axis] if type(axis) == int else axis
        ctx.save_for_backward(a)
        out = max_min_forward(a.data, axis, 'min')
        out = tensorize(out, requires_grad, is_leaf)
        ctx.axis, ctx.out = axis, out.data
        return out

    @staticmethod
    def backward(ctx, grad_output):
        axis, out = ctx.axis, ctx.out
        a, = ctx.saved_tensors
        grad = max_min_backward(grad_output, a.data, out, axis)
        return tensorize(grad)

        

def max_min_forward(a, axis, fun):

    if fun == 'max':
        out = np.amax(a, axis=None if axis is None else tuple(axis), keepdims=True)
    elif fun == 'min':
        out = np.amin(a, axis=None if axis is None else tuple(axis), keepdims=True)
    if axis is not None:
        out = out.reshape([a.shape[i] for i in range(len(a.shape)) if i not in axis])
    return out

def max_min_backward(grad_output, inp, out, axis):
    shape = [1 if axis is None or i in axis else inp.shape[i] for i in range(len(inp.shape))]
    mask = (inp == out.reshape(shape))
    div = mask.sum(axis=None if axis is None else tuple(axis), keepdims=True) 
    return mask * (grad_output.reshape(shape)).data / div
           
class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        seq, dim = args
        grad = [t.requires_grad for t in seq if t is not None]
        requires_grad = f.reduce(lambda x, y: x or y, grad) # requires_grad True if any 
        arr = [t.data for t in seq if t is not None]
        output = tensorize(np.concatenate(arr,dim), requires_grad, not requires_grad)
        shapes = [i.shape for i in seq if i is not None] # shapes for backward
        ctx.cache = dim, shapes
        return output

    @staticmethod
    def backward(ctx,grad_output):
        dim, shapes = ctx.cache 
        sizes = np.cumsum([i[dim] for i in shapes])
        grads = np.array_split(grad_output.data,sizes,dim)
        grads = [tensorize(i) for i in grads]   
        return grads

class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        ctx.save_for_backward(x)
        ctx.indices = indices
        grad = x.requires_grad
        return tensorize(x.data[indices], grad, not grad)

    @staticmethod
    def backward(ctx,grad_output):
        a, = ctx.saved_tensors
        out = np.zeros(a.shape)
        out[ctx.indices] = grad_output.data
        return tensorize(out), None


class AccumulateGrad():
    """Represents node where gradient must be accumulated.
    Args:
        tensor (Tensor): The tensor where the gradients are accumulated in `.grad`
    """

    def __init__(self, tensor):
        self.variable = tensor
        self.next_functions = []  # nodes of current node's parents (this WILL be empty)
        # exists just to be consistent in format with BackwardFunction
        self.function_name = "AccumulateGrad"  # just for convenience lol

    def apply(self, arg):
        """Accumulates gradient provided.
        (Hint: Notice name of function is the same as BackwardFunction's `.apply()`)
        Args:
            arg (Tensor): Gradient to accumulate
        """
        # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = tensor.Tensor(arg.data)
        else:
            self.variable.grad.data += arg.data

        # Some tests to make sure valid grads were stored.
        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        assert shape == grad_shape, (shape, grad_shape)       

def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    # see https://stackoverflow.com/questions/44081007/logsoftmax-stability
    x = predicted
    y = to_one_hot(target,num_classes)

    max = tensor.Tensor(np.max(x.data,axis=1)) #batchsize x 1
    C = (x.T() - max)
    LSM = (C - C.T().exp().sum(axis=1).log()).T()
    Loss = (LSM*y).sum() / tensor.Tensor(-batch_size)

    return Loss

class Dropout(Function):
    @staticmethod
    def forward(ctx, x ,p = 0.5, is_train = False):
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")
        mask = np.random.binomial(n = 1, p = 1 - p, size = x.shape)
        mask = mask/(1-p)
        mask = tensorize(mask, False)
        ctx.save_for_backward(mask)
        
        if is_train:
            x *= mask
        return x
            
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return tensorize(grad_output.data * mask.data)


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(tensor.Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensorize(a, True)

def flatten(x):
    d1, d2 = x.shape[0], np.prod(x.shape[1:])
    return x.reshape((d1, d2))


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        ctx.save_for_backward(x, weight)
        view = asStride(x.data, kernel_size, stride,mode = '1d')
        out = np.einsum('bilk,oik->bol', view, weight.data)
        out += bias.data[None, :, None]
        ctx.cache = (stride, view)
        return tensorize(out, True, False)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, view = ctx.cache
        batch_size, in_channel, input_size = x.shape
        out_channel, in_channel, kernel_size = weight.shape
        batch_size, num_filters, output_length = grad_output.shape
        grad_w = np.einsum('bol,bilk->oik', grad_output.data, view) 
        grad_x = np.zeros((batch_size, in_channel, input_size))
        grad_b = np.einsum('bol->o', grad_output.data)   
        for k in range(output_length):
            X = k % output_length
            iX = X * stride
            grad_x[:, :, iX:iX + kernel_size] += np.einsum('bn, nik->bik', grad_output.data[:, :, X], weight.data) #SLOWER than using tensordot
        grad_x, grad_w, grad_b = map(tensorize, [grad_x, grad_w, grad_b])
        return grad_x, grad_w, grad_b

    
class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        """The forward/backward of a Conv2d Layer in the comp graph.
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width) input data
            weight (Tensor): (out_channel, in_channel, kernel_size, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        out_channel, in_channel, kernel_size, kernel_size = weight.shape
        view = asStride(x.data, (kernel_size,kernel_size), stride)
        out = np.einsum('bihwkl,oikl->bohw', view, weight.data)
        out += bias.data[None, :, None, None]
        ctx.save_for_backward(x, weight)
        ctx.cache = (stride, view)
        return tensorize(out, True, False)
 
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, view = ctx.cache

        batch_size, in_channel, im_height, im_width = x.shape
        num_filters, in_channel, kernel_height, kernel_width = weight.shape
        batch_size, num_filters, output_height, output_width = grad_output.shape
        grad_w = np.einsum('bnhw, bihwyx-> niyx', grad_output.data, view) 
        grad_x = np.zeros((batch_size, in_channel, im_height, im_width))
        grad_b = np.einsum('bnhw-> n', grad_output.data) 

        for k in range(output_height * output_width):
            X, Y = k % output_height, k // output_width
            iX, iY = X * stride, Y * stride # ik,kjyx->ijyx'
            grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.einsum('bn, niyx-> biyx', grad_output.data[:,:,Y,X], weight.data) 

        grad_x = grad_x.reshape((batch_size, in_channel, im_height, im_width))
        dx, dw, db = map(tensorize, [grad_x, grad_w, grad_b])

        return dx, dw, db 


def tensorize(x, grad = False, leaf = False, param = False):
    return tensor.Tensor(x,requires_grad= grad ,is_leaf=leaf, is_parameter= param)
    
def get_conv2d_output_size(input_height, input_width, kernel_size, stride, padding):
    out_h = (input_height - kernel_size + 2 * padding) // stride + 1
    out_w = (input_width - kernel_size + 2 * padding) // stride + 1
    return out_h, out_w

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    return ((input_size - kernel_size) // stride) + 1

class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a max over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        if stride is None:
            stride = 1  
        ctx.save_for_backward(x)
        
        out, pos = pool(x.data, kernel_size, stride,'max',return_max_pos = True)
        ctx.cache = (x.shape, stride, kernel_size, pos)
        out = tensorize(out,True,False)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to xch forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        in_shape, stride, kernel_size, pos = ctx.cache
        out = max_mean_bkwrd(in_shape, grad_output.data, kernel_size, stride, 'max', pos = pos)

        return tensorize(out), None, None

class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a mean over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        if stride is None:
            stride = 1  
        ctx.save_for_backward(x)
        
        out, view = pool(x.data, kernel_size, stride,'mean',return_max_pos = False)
        ctx.cache = (x.shape, stride, kernel_size, view)
        out = tensorize(out, True, False)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to xch forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        in_shape, stride, kernel_size, view = ctx.cache
        out = max_mean_bkwrd(in_shape, grad_output.data, kernel_size, stride, 'mean', view = view)
        return tensorize(out), None, None

def max_mean_bkwrd(in_shape, grad_output, kernel_size, stride, mode, pos = None,view = None):
    
    batch_size, in_channel, in_height, in_width = in_shape
    h_out, w_out = (in_height - kernel_size) // stride + 1, (in_width - kernel_size) // stride + 1
    grad = np.zeros((batch_size, in_channel, in_height, in_width))
    
    if view is not None:
        mask = np.ones(view.shape)
    # 'Pool' back
    for i in range(h_out):
        v_start = i * stride
        v_end = v_start + kernel_size

        for j in range(w_out):
            h_start = j * stride
            h_end = h_start + kernel_size

            if mode == 'max':
                grad[:,:,v_start:v_end,h_start:h_end] += grad_output[:,:,i:i+1,j:j+1] * pos[:,:,i,j,:,:]
            elif mode == 'mean':
                grad[:,:,v_start:v_end,h_start:h_end] += grad_output[:,:,i:i+1,j:j+1] * mask[:,:,i,j,:,:] * (1/(kernel_size * kernel_size))

    return grad

def pool(x, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Strided pooling on 4D data.

    Args:
        x (ndarray): input array to do pooling on the mid 2 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        f (int): pooling kernel size in row/column.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <x> or not. If true, pad <x> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, hi, wi, ci = x.shape
    m, ci,hi, wi = x.shape
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(hi, stride)
        nx = _ceil(wi, stride)
        size = (m, ci,(ny-1)*stride+f, (nx-1)*stride+f)
        x_pad = np.full(size, 0)
        x_pad[... , :hi, :wi] = x
    else:
        x_pad = x[... , :(hi-f)//stride*stride+f, :(wi-f)//stride*stride+f]

    view = asStride(x_pad, (f, f), stride)

    if method == 'max':
        result = np.nanmax(view, axis=(4, 5), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(4, 5), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result, axis=(4, 5))
        return result, pos
    else:
        return result,view


def asStride(arr, sub_shape, stride, mode = '2d'):
    '''Get a strided sub-xrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 4, with shape  m, ci, hi, wi
        sub_shape (tuple): window size: (f1, f2).
        stride (int): stride of windows in both 2nd and 3rd dimensions.
    Returns:
        subs (view): strided window view.

    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape  m, ci, hi, wi, and is transformed
    to a strided view with shape (m, ci, ho, wo, f, f). where:
        m: number of records.
        hi, wi: height and width of input image.
        ci: channels of input image.
        f: kernel size.
    The convolution kernel has shape (f, f, ci, co).
    Then the vectorized 3d convolution can be achieved using either an einsum()
    or a tensordot():

        conv = np.einsum('myxfgc,fgcz->myxz', arr_view, kernel)
        conv = np.tensordot(arr_view, kernel, axes=([3, 4, 5], [0, 1, 2]))

    '''
    if mode == '2d':
        sm, sc, sh, sw = arr.strides
        m, ci, hi, wi = arr.shape
        batch_size, in_channel, im_height, im_width = arr.shape
        f1, f2 = sub_shape
        view_shape = (m, ci,1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2)
        strides = (sm, sc,stride*sh, stride*sw, sh, sw)
    else:
        sm, sc, ss = arr.strides
        batch_size, in_channel, input_size = arr.shape
        kernel_size = sub_shape
        output_length = get_conv1d_output_size(input_size, kernel_size, stride)
        view_shape = (batch_size, in_channel,output_length,kernel_size)
        strides = (sm, sc,stride*ss,ss)
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs


