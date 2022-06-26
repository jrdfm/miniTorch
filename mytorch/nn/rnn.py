import numpy as np
from mytorch import tensor
from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.util import PackedSequence, pack_sequence, unpack_sequence


class RNNUnit(Module):
    '''
    This class defines a single RNN Unit block.

    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        
        super(RNNUnit,self).__init__()
        
        # Initializing parameters
        self.weight_ih = Tensor(np.random.randn(hidden_size,input_size), requires_grad=True, is_parameter=True)
        self.bias_ih   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)
        self.weight_hh = Tensor(np.random.randn(hidden_size,hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hh   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.hidden_size = hidden_size
        
        # Setting the Activation Unit
        if nonlinearity == 'tanh':
            self.act = Tanh()
        elif nonlinearity == 'relu':
            self.act = ReLU()

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)

    def forward(self, input, hidden = None):
        '''
        Args:
            input (Tensor): (effective_batch_size,input_size)
            hidden (Tensor,None): (effective_batch_size,hidden_size)
        Return:
            Tensor: (effective_batch_size,hidden_size)
        '''
        
        # TODO: INSTRUCTIONS
        # Perform matrix operations to construct the intermediary value from input and hidden tensors
        # Apply the activation on the resultant
        # Remeber to handle the case when hidden = None. Construct a tensor of appropriate size, filled with 0s to use as the hidden.
        
        effective_batch_size, input_size = input.shape

        if hidden is None:
            hidden = Tensor(np.zeros((effective_batch_size, self.hidden_size)))

        output = input@(self.weight_ih.T()) + self.bias_ih + hidden @ (self.weight_hh.T()) + self.bias_hh
        return Tanh().forward(output)


class TimeIterator(Module):
    '''
    For a given input this class iterates through time by processing the entire
    seqeunce of timesteps. Can be thought to represent a single layer for a 
    given basic unit which is applied at each time step.
    
    Args:
        basic_unit (Class): RNNUnit or GRUUnit. This class is used to instantiate the unit that will be used to process the inputs
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 

    '''

    def __init__(self, basic_unit, input_size, hidden_size, nonlinearity = 'tanh' ):
        super(TimeIterator,self).__init__()

        # basic_unit can either be RNNUnit or GRUUnit
        self.unit = basic_unit(input_size,hidden_size,nonlinearity)    

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)
    
    def forward(self,input,hidden = None):
        
        '''
        NOTE: Please get a good grasp on util.PackedSequence before attempting this.

        Args:
            input (PackedSequence): input.data is tensor of shape ( total number of timesteps (sum) across all samples in the batch, input_size)
            hidden (Tensor, None): (batch_size, hidden_size)
        Returns:
            PackedSequence: ( total number of timesteps (sum) across all samples in the batch, hidden_size)
            Tensor (batch_size,hidden_size): This is the hidden generated by the last time step for each sample joined together. Samples are ordered in descending order based on number of timesteps. This is a slight deviation from PyTorch.
        '''
        
        # Resolve the PackedSequence into its components
        data, sorted_indices, batch_sizes = input
        # self.sorted_indices,self.batch_sizes
        # print(f'input data {data} \ninput shape {data.shape} sorted_indices {sorted_indices} batch_sizes {batch_sizes} ')
        
        # TODO: INSTRUCTIONS
        # Iterate over appropriate segments of the "data" tensor to pass same timesteps across all samples in the batch simultaneously to the unit for processing.
        # Remeber to account for scenarios when effective_batch_size changes between one iteration to the next
        # print(f'input data \n{data} \ninput shape {data.shape} \nsorted_indices {sorted_indices} batch_sizes {batch_sizes}')
        time_steps = data.split(batch_sizes)
        # [print(f'len timesteps {len(time_steps)} \ntime_steps[{i}] type {type(time_steps[i])} \n{time_steps[i]}\n') for i in range(len(time_steps))]
        # prev_hidden = cur_hidden =  None 
        c = list(np.cumsum(batch_sizes).astype(int))
        hidden_sizes = [y - x for x,y in zip(c,c[1:])]
        out = []
        output_hid = [[] for _ in sorted_indices]
        for i in range(len(time_steps)):
            if hidden:
                hidden = hidden[0:hidden_sizes[i-1]]
            hidden = self.unit.forward(time_steps[i], hidden)
            out.append(hidden)
            # if i == len(time_steps) -1:
            #     output_hid.append(hidden)
        out = tensor.cat(out)
        input.data = out

        # #get final hidden
        unpack = unpack_sequence(input)
        output_ = []
        for t in unpack:
            output_.append(t[-1].unsqueeze())

        # print(f'output_hid len {len(output_hid)} batch_sizes {batch_sizes} sorted_indices {sorted_indices} len ASS {len([[] for _ in sorted_indices])}')

        output_ = [output_[i] for i in input.sorted_indices]
        output_ = tensor.cat(output_)
        # print(f'output_ shape {output_.shape} output_ len {len(output_)} ')
        
        return input,output_


('Implement Forward TimeIterator')


class RNN(TimeIterator):
    '''
    Child class for TimeIterator which appropriately initializes the parent class to construct an RNN.
    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        # TODO: Properly Initialize the RNN class
        super(RNN, self).__init__(RNNUnit, input_size, hidden_size, nonlinearity)

