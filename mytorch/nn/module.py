from mytorch.tensor import Tensor
from mytorch.nn import functional as F
import numpy as np

class Module:
    """Base class (superclass) for all components of an NN.
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    Layer classes and even full Model classes should inherit from this Module.
    Inheritance gives the subclass all the functions/variables below

    NOTE: You shouldn't ever need to instantiate Module() directly."""

    def __init__(self):
        self._submodules = {} # Submodules of the class
        self._parameters = {} # Trainable params in module and its submodules
        self._tensors = {}

        self.is_train = True # Indicator for whether or not model is being trained.

    def train(self):
        """Activates training mode for network component"""
        self.is_train = True

    def eval(self):
        """Activates evaluation mode for network component"""
        self.is_train = False

    def forward(self, *args):
        """Forward pass of the module"""
        raise NotImplementedError("Subclasses of Module must implement forward")

    def is_parameter(self, obj):
        """Checks if input object is a Tensor of trainable params"""
        return isinstance(obj, Tensor) and obj.is_parameter

    def parameters(self):
        """Returns an interator over stored params.
        Includes submodules' params too"""
        self._ensure_is_initialized()
        for name, parameter in self._parameters.items():
            yield parameter
        for name, module in self._submodules.items():
            for parameter in module.parameters():
                yield parameter

    def add_parameter(self, name, value):
        """Stores params"""
        self._ensure_is_initialized()
        self._parameters[name] = value

    def add_module(self, name, value):
        """Stores module and its params"""
        self._ensure_is_initialized()
        self._submodules[name] = value

    def add_tensor(self, name, value) -> None:
        """Stores tensors"""
        self._ensure_is_initialized()
        self._tensors[name] = value

    def __setattr__(self, name, value):
        """Magic method that stores params or modules that you provide"""
        if self.is_parameter(value):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        elif isinstance(value, Tensor):
            self.add_tensor(name, value)


        object.__setattr__(self, name, value)

    def __call__(self, *args):
        """Runs self.forward(args). Google 'python callable classes'"""
        return self.forward(*args)

    def _ensure_is_initialized(self):
        """Ensures that subclass's __init__() method ran super().__init__()"""
        if self.__dict__.get('_submodules') is None:
            raise Exception("Module not intialized. "
                            "Did you forget to call super().__init__()?")
        if self.__dict__.get('_tensors') is None:
            raise Exception("Tensors not initialized. Did not initialize the parent class. \
                             Call super().__init__().")
        if self.__dict__.get('_parameters') is None: 
            raise Exception("Parameters not initialized. Did not initialize the parent class. \
                           Call super().__init__().")
class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        mean = x.sum(axis = 0)/ Tensor(x.shape[0])
        var = ((x - mean) ** Tensor(2)).sum(axis = 0) / Tensor(x.shape[0])

        if self.is_train == True:
            x_hat = (x - mean)/ (var + self.eps)** Tensor(0.5)
            sig = ((x - mean) ** Tensor(2)).sum(axis = 0) / Tensor(x.shape[0] - 1)
            self.running_mean = (Tensor(1) - self.momentum)* self.running_mean + self.momentum* mean
            self.running_var =  (Tensor(1) - self.momentum)* self.running_var + self.momentum* sig
        else:
            x_hat = (x-self.running_mean)/(self.running_var+self.eps)**Tensor(0.5)

        y = self.gamma * x_hat + self.beta

        return y

        
class Dropout(Module):
    """During training, randomly zeroes some input elements with prob `p`.

    This is done using a mask tensor with values sampled from a bernoulli distribution.
    The elements to zero are randomized on every forward call.

    Args:
        p (float): the probability that any neuron output is dropped

    Inherits from:
        Module (mytorch.nn.module.Module)
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.Dropout.apply(x, self.p, self.is_train)


    
class Conv1d(Module):
    """1-dimensional convolutional layer.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for explanations
    and ideas.

    Notes:
        - No, you won't need to implement Conv2d for this homework. Conv2d will be in the bonus assignment.
        - These input args are all that you'll need for this class. You can add more yourself later
          if you want, but not required.

    Args:
        in_channel (int): # channels in input (example: # color channels in image)
        out_channel (int): # channels produced by layer
        kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
        stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        # Initializing weights and bias (not a very good initialization strategy)
        weight = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)

        bias = np.zeros(out_channel)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_size)
        Returns:
            Tensor: (batch_size, out_channel, output_size)
        """
        return F.Conv1d.apply(x, self.weight, self.bias, self.stride)

class Conv2d(Module):
    """2-dimensional convolutional layer.
    Args:
        in_channel (int): # channels in input (example: # color channels in image)
        out_channel (int): # channels produced by layer
        kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
        stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = padding

        # Initializing weights and bias (not a very good initialization strategy)
        # Kaiming init (fan-in) (good init strategy)
        # bound = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        # weight = np.random.uniform(-bound, bound, size=(out_channel, in_channel,
        # kernel_size, kernel_size))
        
        # self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        # bias = np.random.uniform(-bound, bound, size=(out_channel,))
        # self.bias = Tensor(bias, requires_grad=True, is_parameter=True)


        self.weight = Tensor(1e-3 * np.random.randn(out_channel, in_channel, kernel_size, kernel_size), requires_grad=True, is_parameter=True)
        self.bias = Tensor(np.zeros(out_channel), requires_grad=True, is_parameter=True)

    def forward (self, x):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_size)
        Returns:
            Tensor: (batch_size, out_channel, output_size)
        """
        return F.Conv2d.apply(x, self.weight, self.bias, self.stride,self.padding)     


        

class Flatten(Module):
    """Layer that flattens all dimensions for each observation in a batch

    >>> x = torch.randn(4, 3, 2) # batch of 4 observations, each sized (3, 2)
    >>> x
    tensor([[[ 0.8816,  0.9773],
             [-0.1246, -0.1373],
             [-0.1889,  1.6222]],

            [[-0.9503, -0.8294],
             [ 0.8900, -1.2003],
             [-0.9701, -0.4436]],

            [[ 1.7809, -1.2312],
             [ 1.0769,  0.6283],
             [ 0.4997, -1.7876]],

            [[-0.5303,  0.3655],
             [-0.7496,  0.6935],
             [-0.8173,  0.4346]]])
    >>> layer = Flatten()
    >>> out = layer(x)
    >>> out
    tensor([[ 0.8816,  0.9773, -0.1246, -0.1373, -0.1889,  1.6222],
            [-0.9503, -0.8294,  0.8900, -1.2003, -0.9701, -0.4436],
            [ 1.7809, -1.2312,  1.0769,  0.6283,  0.4997, -1.7876],
            [-0.5303,  0.3655, -0.7496,  0.6935, -0.8173,  0.4346]])
    >>> out.shape
    torch.size([4, 6]) # batch of 4 observations, each flattened into 1d array size (6,)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, dim_2, dim_3, ...) arbitrary number of dims after batch_size
        Returns:
            out (Tensor): (batch_size, dim_2 * dim_3 * ...) batch_size, then all other dims flattened
        """
        raise Exception("TODO! One line of code. See writeup for hint.")




class MaxPool1d(Module):
    """Performs a max pooling operation after a 1d convolution
        
        Args:
            kernel_size (int): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x:Tensor) -> Tensor:
        """Performs a max pooling operation after a 1d convolution

        Args:
            x (Tensor): (batch_size, channel, in_width, in_height)

        Returns:
            Tensor: (batch_size, channel, out_width, out_height)
        """
        out = x.max_pool1d(self.kernel_size)
        out.name = 'mpool1d_res'
        return out


class AvgPool1d(Module):
    """Performs an average pooling operation after a 1d convolution
        
        Args:
            kernel_size (int): Kernel size
            stride (int): Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size:int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x:Tensor) -> Tensor:
        """Performs an average pooling operation after a 1d convolution

        Args:
            x (Tensor): (batch_size, channel, in_width, in_height)

        Returns:
            Tensor: (batch_size, channel, out_width, out_height)
        """
        out = x.avg_pool1d(self.kernel_size)
        out.name = 'avgpool1d_res'
        return out


class MaxPool2d(Module):
    """Performs a max pooling operation after a 2d convolution
    
        Args:
            kernel_size : Kernel size
            stride : Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size, stride = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size 
        self.stride = stride 
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, channel, in_width, in_height)
            
            Returns:
                Tensor: (batch_size, channel, out_width, out_height)
        """
        
        out = x.max_pool2d(self.kernel_size, self.stride)
        out.name = 'mpool2d_res'
        return out


class AvgPool2d(Module):
    """Performs an average pooling operation after a 2d convolution

        Args:
            kernel_size : Kernel size
            stride : Stride
        
        Inherits from:
            Module (nn.module.Module)
    """
    def __init__(self, kernel_size, stride = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size 
        self.stride = stride 
    
    def forward(self, x:Tensor) -> Tensor:
        """
            Args:
                x (Tensor): (batch_size, channel, in_width, in_height)
            
            Returns:
                Tensor: (batch_size, channel, out_width, out_channel)
        """
        out = x.avg_pool2d(self.kernel_size, self.stride)
        out.name = 'avgpool2d_res'
        return out

class Linear(Module):
    """A linear layer (aka 'fully-connected' or 'dense' layer)

    >>> layer = Linear(2,3)
    >>> layer(Tensor.ones(10,2)) # (batch_size, in_features)
    <some tensor output with size (batch_size, out_features)>

    Args:
        in_features (int): # dims in input
                           (i.e. # of inputs to each neuron)
        out_features (int): # dims of output
                           (i.e. # of neurons)

    Inherits from:
        Module (mytorch.nn.module.Module)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Randomly initializing layer weights
        k = 1 / in_features
        weight = k * (np.random.rand(out_features, in_features) - 0.5)
        bias = k * (np.random.rand(out_features) - 0.5)
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, in_features)
        Returns:
            Tensor: (batch_size, out_features)
        """
        # check that the input is a tensor
                # if not type(x).__name__ == 'Tensor':
                # raise Exception("Only dropout for tensors is supported")
        if not (type(x).__name__ == 'Tensor' or type(self.weight).__name__ == 'Tensor'):
            raise Exception(f"X must be Tensor. Got {type(x)}")
        output = x @ self.weight.T() + self.bias
        return output


class Sequential(Module):
    """Passes input data through stored layers, in order

    >>> model = Sequential(Linear(2,3), ReLU())
    >>> model(x)
    <output after linear then relu>

    Inherits from:
        Module (nn.module.Module)
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        # iterate through args provided and store them
        for idx, l in enumerate(self.layers):
            self.add_module(str(idx), l)

    def __iter__(self):
        """Enables list-like iteration through layers"""
        yield from self.layers

    def __getitem__(self, idx):
        """Enables list-like indexing for layers"""
        return self.layers[idx]

    def train(self):
        """Sets this object and all trainable modules within to train mode"""
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self):
        """Sets this object and all trainable modules within to eval mode"""
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x):
        """Passes input data through each layer in order
        Args:
            x (Tensor): Input data
        Returns:
            Tensor: Output after passing through layers
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


class CrossEntropyLoss(Module):
    """The XELoss function.
    This class is for human use; just calls function in nn.functional.
    Does not need args to initialize.

    >>> criterion = CrossEntropyLoss()
    >>> criterion(outputs, labels)
    3.241
    """
    def __init__(self):
        pass

    def forward(self, predicted, target):
        """
        Args:
            predicted (Tensor): (batch_size, num_classes)
            target (Tensor): (batch_size,)

        Returns:
            Tensor: loss, stored as a float in a tensor
        """
        # Simply calls nn.functional.cross_entropy
        # If you implement your own Function subclass you may need to modify this"""
        return F.cross_entropy(predicted, target)
