from matplotlib.pyplot import axes
import numpy as np
from mytorch import tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
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
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
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
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
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
        # dL/da = dout/da * dL/dout
        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

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
        return tensor.Tensor(grad), None, None

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
        # dL/da = dout/da * dL/dout
        grad_a = grad_output.data / b.data
        # dL/db = dout/db * dL/dout
        grad_b = (-a.data * grad_output.data) / (b.data ** 2)

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

# TODO: Implement more Functions below

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        if (type(b).__name__ == 'int'):
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
        out_channel, in_channel, kernel_size, kernel_size= weight.shape

        n, c, h, w = x.shape
        out_h, out_w = get_conv2d_output_size(h, w, kernel_size, stride, padding)

        cols = to_cols(x.data, weight.data, stride,out_h, out_w)

        out = np.einsum('bihwkl,oikl->bohw', cols, weight.data)

        out += bias.data[None, :, None, None]
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.cache = (stride, cols)
        return to_tensor(out, True, False)

 
    @staticmethod
    def backward(ctx, grad_output):
        # weight, stride, input, input_reshaped = ctx.saved_tensors

        x, weight = ctx.saved_tensors
        stride, input_reshaped = ctx.cache

        batch_size, in_channel, im_height, im_width = x.shape
        num_filters, _, kernel_height, kernel_width = weight.shape
        _, _, output_height, output_width = grad_output.shape
        
        grad_w = np.einsum('ikYX, ijYXyx -> kjyx', grad_output.data, input_reshaped) 
        # grad_w = np.tensordot(grad_output.data, input_reshaped, axes=[(0,2,3),(0,2,3)])

        grad_x = np.zeros((batch_size, in_channel, im_height, im_width), dtype=grad_output.data.dtype)

        for k in range(output_height * output_width):
            X, Y = k % output_width, k // output_width
            iX, iY = X * stride, Y * stride

            grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.einsum('ik,kjyx->ijyx', grad_output.data[:,:,Y,X], weight.data) 
            # SLOWER than using tensordot
            # grad_x[:,:, iY:iY+kernel_height, iX:iX+kernel_width] += np.tensordot(grad_output.data[:,:,Y,X], weight.data, axes=[(1), (0)])

        grad_x = grad_x.reshape((batch_size, in_channel, im_height, im_width))

        grad_b = np.sum(grad_output.data, axis= (0, 2, 3)) 
        dx, dw, db = map(to_tensor, [grad_x, grad_w, grad_b])

        return dx, dw, db 


def to_cols(x, w, stride,output_height, output_width):
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


def to_tensor(x, grad = False, leaf = False, param = False):
    return tensor.Tensor(x,requires_grad= grad ,is_leaf=leaf, is_parameter= param)
    

def get_conv2d_output_size(input_height, input_width, kernel_size, stride, padding):
    out_h = (input_height - kernel_size + 2 * padding) // stride + 1
    out_w = (input_width - kernel_size + 2 * padding) // stride + 1
    return out_h, out_w

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return int(((input_size - kernel_size) / stride) + 1)

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
        raise Exception("TODO: Finish MaxPool2d(Function).forward() for hw2 bonus")


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        raise Exception("TODO: Finish MaxPool2d(Function).backward() for hw2 bonus")



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
        raise Exception("TODO: Finish AvgPool2d(Function).forward() for hw2 bonus")


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        raise Exception("TODO: Finish AvgPool2d(Function).backward() for hw2 bonus")
