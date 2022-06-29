import numpy as np

from mytorch.nn.activations import ReLU
from mytorch.nn.functional import tensorize
from mytorch.nn.module import *

class CNN_:
    def convert_weights(self, weight, parm):
        out_ch, in_ch, ker_sz = parm
        return tensorize(np.reshape(weight, (out_ch, ker_sz, in_ch)).transpose(0, 2, 1), True, True, True)

class CNN_SimpleScanningMLP(CNN_):
    """Question 2.1: CNN as a Simple Scanning MLP

    Complete the indicated methods to convert the linear MLP described
    in the assignment sheet into a Simple 1D CNN.
    """
    def __init__(self):
        # TODO: Initialize Conv1d layers with appropriate params (this is the hard part)
        # For reference, here's the arguments for Conv1d:
        #            Conv1d(in_channel, out_channel, kernel_size, stride)
        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)

        # TODO: Initialize Sequential object with layers based on the MLP architecture.
        # Note: Besides using Conv1d instead of Linear, there is a slight difference in layers.
        #       What's the difference and why?
        self.layers = Sequential(self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten())

    def init_weights(self, weights):
        """Converts the given 3 weight matrices of the linear MLP into the weights of the Conv layers.
        Args:
            weights (tuple(np.array)): shapes ((8, 192), (16, 8), (4, 16))
                                       Think of each as a Linear.weight.data, shaped (out_features, in_features)
        """
        # TODO: Convert the linear weight arrays into Conv1d weight tensors
        # Make sure to not add nodes to the comp graph!
        w1, w2, w3 = weights # Here, we've unpacked them into separate arrays for you.
        
        # Assume the Conv1d weight tensors are already initialized with the params that you specified in __init__().
        # Your job now is to replace those weights with the MLP's weights.
        
        # Tip: You can automatically retrieve the Conv1d params like so:
        #      ex) self.conv1.out_channel, self.conv1.kernel_size, self.conv1.in_channel

        # Set the weight tensors with your converted MLP weights
        parms = [(i.out_channel, i.in_channel, i.kernel_size) for i in [self.conv1, self.conv2, self.conv3]]
        self.conv1.weight, self.conv2.weight, self.conv3.weight = map(self.convert_weights, [w1, w2, w3], parms)

    def forward(self, x):
        """Do not modify this method

        Args:
            x (Tensor): (batch_size, in_channel, in_features)
        Returns:
            Tensor: (batch_size, out_channel, out_features)
        """
        return self.layers(x)

    def __call__(self, x):
        """Do not modify this method"""
        return self.forward(x)
    
class CNN_DistributedScanningMLP(CNN_):
    """Question 2.2: CNN as a Distributed Scanning MLP

    Complete the indicated methods to convert the linear MLP described
    in the assignment sheet into a Distributed 1D CNN."""
    def __init__(self):
        # TODO: Initialize Conv1d layers
        # For reference, here's the arguments for Conv1d:
        #            Conv1d(in_channel, out_channel, kernel_size, stride)
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)

        # TODO: Initialize Sequential object
        self.layers = Sequential(self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten())

    def __call__(self, x):
        """Do not modify this method"""
        return self.forward(x)

    def init_weights(self, weights):
        """Use the 3 weight matrices of linear MLP to init the weights of the CNN.
        Args:
            weights (tuple(np.array)): shapes ((8, 192), (16, 8), (4, 16))
                                       Think of each as a Linear.weight.data, shaped (out_features, in_features)
        """
        w1, w2, w3 = weights

        # TODO: Convert the linear weights into Conv1d weights
        # Make sure to not add nodes to the comp graph! idk what this means
        w1 = w1[0:2:,0:48]
        w2 = w2[0:8:,0:4]

        parms = [(i.out_channel, i.in_channel, i.kernel_size) for i in [self.conv1, self.conv2, self.conv3]]
        self.conv1.weight, self.conv2.weight, self.conv3.weight = map(self.convert_weights, [w1, w2, w3], parms)

    def forward(self, x):
        """Already completed for you.
        Args:
            x (Tensor): (batch_size, in_channel, in_features)
        Returns:
            out (Tensor): (batch_size, out_channel, out_features)
        """
        return self.layers(x)
