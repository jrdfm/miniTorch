"""This file contains new code for hw2 bonus that you should copy+paste to the appropriate files.

We'll tell you where each method/class belongs."""

# ---------------------------------
# nn/functional.py
# ---------------------------------
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
