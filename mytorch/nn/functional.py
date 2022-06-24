import numpy as np
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
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        print(type(grad_output))
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return tensor.Tensor(grad_output.data / a.data)

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

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should xch the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors & of equal shape: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a ,b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data,requires_grad = requires_grad,is_leaf = not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data

        # the order of gradients returned should xch the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/grad_output = dout/grad_output * dL/dout
        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data

        # the order of gradients returned should xch the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims = None):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only Sum of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
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
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensorize(grad), None, None



class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
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
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

# TODO: Implement more Functions below

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        if (type(b).__name__ == 'int' or type(b).__name__ == 'float'):
            b = tensor.Tensor(np.array(b))
        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data**b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #b = Tensor.ones(*a.shape) * b
        grad_a =  b.data * (a.data ** (b.data - 1)) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        return grad_a,None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that arg is a tensor
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("arg must be a Tensor: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad 
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_a =  np.exp(a.data) * grad_output.data

        return tensor.Tensor(grad_a),None

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
                    a.data.shape[1] != b.data.shape[0]:
                    raise Exception("Both args must be Tensors & of same shape: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a ,b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data @ b.data, requires_grad = requires_grad,is_leaf = not requires_grad)
        return c
    @staticmethod
    def backward(ctx, grad_output):
        a, b  = ctx.saved_tensors
        grad_a = tensor.Tensor(grad_output.data @ b.data.T)
        grad_b = tensor.Tensor(a.data.T @ grad_output.data)

        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that arg is a tensor
        if not (type(a).__name__ == 'Tensor'):
            raise Exception("arg must be a Tensor: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad 
        c = tensor.Tensor(np.where(a.data > 0,a.data, 0),requires_grad= requires_grad,
                    is_leaf= not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a,  = ctx.saved_tensors
        grad_a = tensor.Tensor(np.where(a.data > 0, 1, 0)* grad_output.data)
        
        return grad_a
class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad)
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad)
       
class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Sqrt must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data,requires_grad = requires_grad,
                    is_leaf = not requires_grad))
        return c
            
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        output = tensor.Tensor(1/2 *(a.data**(-1/2)) * grad_output.data)
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

        
class Slice(Function):
    @staticmethod
    def forward(ctx, a, indices=None):
        ctx.shape, ctx.indices = a.shape, indices
        requires_grad, is_leaf  = a.requires_grad, not a.requires_grad
        out_data = inner_slice(a.data, indices)
        out = tensorize(out_data, requires_grad, is_leaf)
        out.children = [a]
        out.op = 'slice'
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        shape, fwd_indices = ctx.shape, ctx.indices
        indices = [(0 - p[0], grad_output.shape[i] + (shape[i] - p[1])) for i, p in enumerate(fwd_indices)]
        grad = inner_slice(grad_output, indices)
        return tensorize(grad), None

def inner_slice(a, indices):
    """
        Helper function to slice a Tensor

        Args:
            a (np.ndarray): array to slice
            indices (list): list of indices 
        
        ..note: Length must xch the number of dimensions of x
    """
    padding = [(max(0, -p[0]), max(0, p[1]-a.shape[i])) for i, p in enumerate(indices)]
    a = np.pad(a, padding, mode="constant")
    slices = [(p[0]+padding[i][0], p[1]+padding[i][0]) for i, p in enumerate(indices)]
    return a[tuple([slice(x[0], x[1], None) for x in slices])]


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
        mask = tensor.Tensor(mask, requires_grad=False)
        ctx.save_for_backward(mask)
        
        if is_train:
            x *= mask
        return x
            
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return tensor.Tensor(grad_output.data * mask.data)


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
    return tensor.Tensor(a, requires_grad = True)


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
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        
        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        # output_size = get_conv1d_output_size(None, None, None)

        # TODO: Initialize output with correct size
        # out = np.zeros(())
        
        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        
        # TODO: Put output into tensor with correct settings and return 
        raise NotImplementedError("Implement functional.Conv1d.forward()!")
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        raise NotImplementedError("Implement functional.Conv1d.backward()!")

    
class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        """The forward/backward of a Conv2d Layer in the comp graph.
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        # batch_size, in_channel, input_height, input_width = x.shape
        out_channel, in_channel, kernel_size, kernel_size = weight.shape

        n, c, h, w = x.shape
        out_h, out_w = get_conv2d_output_size(h, w, kernel_size, stride, padding)

        cols = window(x.data, weight.data, stride, out_h, out_w)
        out = np.einsum('bihwkl,oikl->bohw', cols, weight.data)

        out += bias.data[None, :, None, None]
        ctx.save_for_backward(x, weight)
        ctx.cache = (stride, cols)
        return tensorize(out, True, False)

 
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, input_reshaped = ctx.cache

        batch_size, in_channel, im_height, im_width = x.shape
        num_filters, _, kernel_height, kernel_width = weight.shape
        _, _, output_height, output_width = grad_output.shape

        grad_w = np.einsum('ikYX, ijYXyx -> kjyx', grad_output.data, input_reshaped) 
        # grad_w = np.tensordot(grad_output.data, input_reshaped, axes=[(0,2,3),(0,2,3)])
        grad_x = np.zeros((batch_size, in_channel, im_height, im_width), dtype=grad_output.data.dtype)

        for k in range(output_height * output_width):
            X, Y = k % output_height, k // output_width
            iX, iY = X * stride, Y * stride
            grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.einsum('ik,kjyx->ijyx', grad_output.data[:,:,Y,X], weight.data) 
            # SLOWER than using tensordot
            # grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.tensordot(grad_output.data[:,:,Y,X], weight.data, axes=[(1), (0)])

        grad_x = grad_x.reshape((batch_size, in_channel, im_height, im_width))

        grad_b = np.sum(grad_output.data, axis= (0, 2, 3)) 
        dx, dw, db = map(tensorize, [grad_x, grad_w, grad_b])

        return dx, dw, db 


def window(x, w, stride,output_height, output_width):
        num_filters, _, kernel_height, kernel_width = w.shape
        batch_size, in_channel, im_height, im_width = x.shape
        strides = (im_height * im_width, im_width, 1, in_channel * im_height * im_height, stride * im_width, stride)
        strides = x.itemsize * np.array(strides)

        cols = np.lib.stride_tricks.as_strided(
            x=x,
            shape=(in_channel, kernel_height, kernel_width, batch_size, output_height, output_width),
            strides=strides,
            writeable=False
        )
        cols = cols.transpose(3, 0, 4, 5, 1, 2)
        return cols


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
    return ((input_size - kernel_size) / stride) + 1

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
        # x, = ctx.saved_tensors
        in_shape, stride, kernel_size, pos = ctx.cache
        # out = unpool(grad_output.data, pos, ori_shape, stride)
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


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-xrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 4, with shape (m, hi, wi, ci).
        sub_shape (tuple): window size: (f1, f2).
        stride (int): stride of windows in both 2nd and 3rd dimensions.
    Returns:
        subs (view): strided window view.

    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape (m, hi, wi, ci), and is transformed
    to a strided view with shape (m, ho, wo, f, f, ci). where:
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
    sm, sc, sh, sw= arr.strides
    m, ci, hi, wi = arr.shape
    f1, f2 = sub_shape

    view_shape = (m, ci,1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2)
    strides = (sm, sc,stride*sh, stride*sw, sh, sw)

    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs

