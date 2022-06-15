import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('autograder')
from helpers import *

sys.path.append('./')
from mytorch.nn.conv import *
from mytorch.optim.sgd import SGD
from mytorch.nn.sequential import Sequential
from mytorch.nn.linear import Linear
from mytorch.nn.loss import *
from mytorch.nn.activations import *
from mytorch.tensor import Tensor


def test_conv2d_forward():
    np.random.seed(11785)

    # Tests CNNs with 1 to 3 Conv1d layers.
    for num_layers in range(1, 3):
        # Determine random params
        in_c = np.random.randint(5, 15)
        channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
        kernel = [np.random.randint(3, 7) for i in range(num_layers)]
        stride = [np.random.randint(3, 5) for i in range(num_layers)]
        
        # Init model with above params
        test_layers = [Conv2d(channels[i], channels[i + 1], kernel[i], stride[i])
                        for i in range(num_layers)]
        test_model = Sequential(*test_layers)
        
        # Run just the model forward
        test_forward(test_model)
    return True


def test_conv2d_backward():
    np.random.seed(11785)

    for num_layers in range(1, 3):
        in_c = np.random.randint(5, 15)
        channels = [np.random.randint(5, 15) for i in range(num_layers + 1)]
        kernel = [np.random.randint(3,7) for i in range(num_layers)]
        stride = [np.random.randint(3,5) for i in range(num_layers)]

        test_layers = [Conv2d(channels[i], channels[i + 1], kernel[i], stride[i])
                        for i in range(num_layers)]
        test_model = Sequential(*test_layers)
        mytorch_optimizer = SGD(test_model.parameters())
        test_step(test_model, mytorch_optimizer, 5)

    return True



##############################
# Utilities for testing MLPs #
##############################

def test_forward(mytorch_model, mytorch_criterion = None, batch_size = None, width = None):
    """
    Tests forward, printing whether a mismatch occurs in forward or backwards.

    Returns whether the test succeeded.
    """
    pytorch_model = get_same_pytorch_mlp(mytorch_model)
    batch_size = batch_size if not batch_size is None else np.random.randint(1, 4)
    width = width if not width is None else np.random.randint(60, 80)
    x, y = generate_dataset_for_mytorch_model(mytorch_model, width, batch_size)
    pytorch_criterion = get_same_pytorch_criterion(mytorch_criterion)

    forward_passed, _ = forward_(mytorch_model, mytorch_criterion,
                                 pytorch_model, pytorch_criterion, x, y)
    if not forward_passed:
        print("Forward failed")
        return False

    return True


def test_step(mytorch_model, mytorch_optimizer, steps,
              mytorch_criterion=None, batch_size=None,
              pytorch_model=None, width=None):
    """
    Tests subsequent forward, back, and update operations, printing whether
    a mismatch occurs in forward or backwards.

    Returns whether the test succeeded.
    """
    if pytorch_model is None:
        pytorch_model = get_same_pytorch_mlp(mytorch_model)
    pytorch_optimizer = get_same_pytorch_optimizer(mytorch_optimizer, pytorch_model)
    pytorch_criterion = get_same_pytorch_criterion(mytorch_criterion)
    batch_size = batch_size if not batch_size is None else np.random.randint(1, 4)
    width = width if not width is None else np.random.randint(60, 80)
    
    x, y = generate_dataset_for_mytorch_model(mytorch_model, width, batch_size)

    displayed_error = False # to make sure annoying error only displayed once and not over and over
    for s in range(steps):
        pytorch_optimizer.zero_grad()
        mytorch_optimizer.zero_grad()

        forward_passed, (mx, my, px, py) = \
                forward_(mytorch_model, mytorch_criterion,
                         pytorch_model, pytorch_criterion, x, y)
        if not forward_passed:
            print("Forward failed")
            return False

        backward_passed = backward_(mx, my, mytorch_model, px, py, pytorch_model)
        if not backward_passed:
            print("Backward failed")
            return False

        # Check that the output and input tensor have correct settings and gradient settings
        correct_output_settings = check_operation_output_settings(my, mx,
                                                                  backpropped=True)
        if not displayed_error and not correct_output_settings:
            displayed_error = True

        pytorch_optimizer.step()
        mytorch_optimizer.step()

        # Check that model settings are still fine after stepping
        check_model_param_settings(mytorch_model)

    return True


def get_same_pytorch_mlp(mytorch_model):
    """
    Returns a pytorch Sequential model matching the given mytorch mlp, with
    weights copied over.
    """
    layers = []
    for l in mytorch_model.layers:
        if isinstance(l, Linear):
            layers.append(nn.Linear(l.in_features, l.out_features))
            layers[-1].weight = nn.Parameter(torch.tensor(l.weight.data))
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data))
        elif isinstance(l, ReLU):
            layers.append(nn.ReLU())
        elif isinstance(l, Conv1d):
            layers.append(nn.Conv1d(l.in_channel, l.out_channel,
                                    kernel_size = l.kernel_size, stride = l.stride))
            layers[-1].weight = nn.Parameter(torch.tensor(l.weight.data))
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data))
        elif isinstance(l, Conv2d):
            layers.append(nn.Conv2d(l.in_channel, l.out_channel,
                                    kernel_size = l.kernel_size, stride = l.stride))
            layers[-1].weight = nn.Parameter(torch.tensor(l.weight.data))
            layers[-1].bias = nn.Parameter(torch.tensor(l.bias.data))
        else:
            raise Exception("Unrecognized layer in mytorch model")
    pytorch_model = nn.Sequential(*layers)
    return pytorch_model


def get_same_pytorch_optimizer(mytorch_optimizer, pytorch_mlp):
    """
    Returns a pytorch optimizer matching the given mytorch optimizer, except
    with the pytorch mlp parameters, instead of the parametesr of the mytorch
    mlp
    """
    lr = mytorch_optimizer.lr
    momentum = mytorch_optimizer.momentum
    return torch.optim.SGD(pytorch_mlp.parameters(), lr = lr, momentum = momentum)

def get_same_pytorch_criterion(mytorch_criterion):
    """
    Returns a pytorch criterion matching the given mytorch optimizer
    """
    if mytorch_criterion is None:
        return None
    return nn.CrossEntropyLoss()

def generate_dataset_for_mytorch_model(mytorch_model, width, batch_size):
    """
    Generates a fake dataset to test on.

    Returns x: ndarray (batch_size, in_features),
            y: ndarray (batch_size,)
    where in_features is the input dim of the mytorch_model, and out_features
    is the output dim.
    """
    in_features = get_mytorch_model_input_features(mytorch_model)
    out_features = get_mytorch_model_output_features(mytorch_model)
    x = np.random.randn(batch_size, in_features, width, width)
    y = np.random.randint(out_features, size = (batch_size,))
    return x, y

def generate_cnn_dataset_for_mytorch_model(mytorch_model, batch_size):
    """
    Generates a fake dataset to test on.

    Returns x: ndarray (batch_size, in_features),
            y: ndarray (batch_size,)
    where in_features is the input dim of the mytorch_model, and out_features
    is the output dim.
    """
    width =  60
    out_features = 10
    x = np.random.randn(batch_size, 24, width)
    y = np.random.randint(out_features, size = (batch_size,))
    return x, y

def forward_(mytorch_model, mytorch_criterion, pytorch_model,
             pytorch_criterion, x, y):
    """
    Calls forward on both mytorch and pytorch models.

    x: ndrray (batch_size, in_features)
    y: ndrray (batch_size,)

    Returns (passed, (mytorch x, mytorch y, pytorch x, pytorch y)),
    where passed is whether the test passed

    """
    # forward
    pytorch_x = Variable(torch.tensor(x).double(), requires_grad=True)
    pytorch_y = pytorch_model(pytorch_x)
    if not pytorch_criterion is None:
        pytorch_y = pytorch_criterion(pytorch_y, torch.LongTensor(y))
    mytorch_x = Tensor(x, requires_grad=True)
    mytorch_y = mytorch_model(mytorch_x)
    if not mytorch_criterion is None:
        mytorch_y = mytorch_criterion(mytorch_y, Tensor(y))

    # check that output tensor is correctly configured
    check_operation_output_settings(mytorch_y, mytorch_x, b=None, backpropped=False)

    # check that model is still correctly configured
    check_model_param_settings(mytorch_model)

    # forward check
    if not assertions_all(mytorch_y.data, pytorch_y.detach().numpy(), 'y'):
        return False, (mytorch_x, mytorch_y, pytorch_x, pytorch_y)

    return True, (mytorch_x, mytorch_y, pytorch_x, pytorch_y)


def backward_(mytorch_x, mytorch_y, mytorch_model, pytorch_x, pytorch_y, pytorch_model):
    """
    Calls backward on both mytorch and pytorch outputs, and returns whether
    computed gradients match.
    """
    mytorch_y.backward()
    pytorch_y.sum().backward()

    # check that model is still correctly configured
    check_model_param_settings(mytorch_model)

    return check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model)


def check_gradients(mytorch_x, pytorch_x, mytorch_model, pytorch_model):
    """
    Checks computed gradients, assuming forward has already occured.

    Checked gradients are the gradients of linear weights and biases, and the
    gradient of the input.
    """

    if not assertions_all(mytorch_x.grad.data, pytorch_x.grad.detach().numpy(), 'dx'):
        return False
    mytorch_linear_layers = get_mytorch_conv_layers(mytorch_model)
    pytorch_linear_layers = get_pytorch_conv_layers(pytorch_model)
    for mytorch_linear, pytorch_linear in zip(mytorch_linear_layers, pytorch_linear_layers):
        pytorch_dW = pytorch_linear.weight.grad.detach().numpy()
        pytorch_db = pytorch_linear.bias.grad.detach().numpy()
        mytorch_dW = mytorch_linear.weight.grad.data
        mytorch_db = mytorch_linear.bias.grad.data

        if not assertions_all(mytorch_dW, pytorch_dW, 'dW'): return False
        if not assertions_all(mytorch_db, pytorch_db, 'db'): return False
    return True


def get_mytorch_model_input_features(mytorch_model):
    """
    Returns in_features for the first linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_conv_layers(mytorch_model)[0].in_channel

def get_mytorch_model_output_features(mytorch_model):
    """
    Returns out_features for the last linear layer of a mytorch
    Sequential model.
    """
    return get_mytorch_conv_layers(mytorch_model)[-1].out_channel

def get_mytorch_conv_layers(mytorch_model):
    """
    Returns a list of linear layers for a mytorch model.
    """
    return list(filter(lambda x: isinstance(x, Conv2d), mytorch_model.layers))

def get_pytorch_conv_layers(pytorch_model):
    """
    Returns a list of linear layers for a pytorch model.
    """
    return list(filter(lambda x: isinstance(x, nn.Conv2d), pytorch_model))


