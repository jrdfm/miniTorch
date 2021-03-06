import os
import sys
import numpy as np

# from mytorch.nn.conv import Conv1d, Flatten
from mytorch.nn.functional import get_conv1d_output_size
# from mytorch.nn.linear import Linear
from mytorch.nn.module import *
# from mytorch.nn.sequential import Sequential


class CNN(Module):
    """A simple convolutional neural network with the architecture described in Section 3.
    
    You'll probably want to implement `get_final_conv_output_size()` on the
        bottom of this file as well.
    """
    def __init__(self):
        super().__init__()
        
        # You'll need these constants for the first layer
        first_input_size = 60 # The width of the input to the first convolutional layer
        fst_in_channel = 24 # The number of channels input into the first layer
        fst_in_channel, fst_out_channel = 24, 56 
        sec_in_channel, sec_out_channel = fst_out_channel, 28
        trd_in_channel, trd_out_channel = sec_out_channel, 14
        fst_kernel_size, sec_kernel_size, trd_kernel_size = 5, 6, 2
        fst_stride, sec_stride = 1, 2

        # TODO: initialize all layers EXCEPT the last linear layer
        layers = [
            Conv1d(fst_in_channel, fst_out_channel, fst_kernel_size, fst_stride),
            Tanh(),
            Conv1d(sec_in_channel, sec_out_channel, sec_kernel_size, sec_stride),
            ReLU(),
            Conv1d(trd_in_channel, trd_out_channel, trd_kernel_size, sec_stride),
            Sigmoid(),
            Flatten(),
        ]
        
        # TODO: Iterate through the conv layers and calculate the final output size
        in_features = get_final_conv_output_size(layers , first_input_size)
        # TODO: Append the linear layer with the correct size onto `layers`
        layers.append(Linear(in_features, 10))
        # TODO: Put the layers into a Sequential
        self.layers = Sequential(*layers)
        

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_feature)
        """
        # Already completed for you. Passes data through all layers in order.
        return self.layers(x)


def get_final_conv_output_size(layers, input_size):
    """Calculates how the last dimension of the data will change throughout a CNN model
    
    Note that this is the final output size BEFORE the flatten.
    
    Note:
        - You can modify this function to use in HW2P2.
        - If you do, consider making these changes:
            - Change `layers` variable to `model` (a subclass of `Module`),
                and iterate through its submodules
                - Make a toy model in torch first and learn how to do this
            - Modify calculations to account for other layer types (like `Linear` or `Flatten`)
            - Change `get_conv1d_output_size()` to account for stride and padding (see its comments)
    
    Args:
        layers (list(Module)): List of Conv1d layers, activations, and flatten layers
        input_size (int): input_size of x, the input data 
    """
    # Hint, you may find the function `isinstance()` to be useful.
    output_size = 0
    for model in layers:
        if isinstance(model, Conv1d):
            input_size = get_conv1d_output_size(input_size, model.kernel_size, model.stride)
            output_size = input_size * model.out_channel
    return output_size
