import pdb
from mytorch import tensor



def backward(grad_fn, grad_of_outputs):
    """Recursive DFS that traverses comp graph, handing back gradients as it goes.
    Args:
        grad_fn (BackwardFunction or AccumulateGrad): Current node type from
                                                      parent's `.next_functions`
        grad_of_output (Tensor): Gradient of the final node w.r.t. current output
    Returns:
        No return statement needed.
    """
    # Calculate gradients of final node w.r.t. to the current nodes parents
    if grad_fn is not None:
        new_grad = grad_fn.apply(grad_of_outputs)
        # Pass gradient onto current node's parents (recursive DFS)
        for i in range(len(grad_fn.next_functions)):
            nxt = grad_fn.next_functions[i]
            if nxt is not None:
                if type(new_grad) == tensor.Tensor:
                    grad = new_grad
                else:
                    grad = new_grad[i]
                backward(nxt, grad)
    
class Function:
    """Superclass for linking nodes to the computational graph.
    Operations in `functional.py` should inherit from this"""

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("All subclasses must implement backward")

    @classmethod
    def apply(cls, *args):
        """Runs forward of subclass and links node to the comp graph.
        Args: cls (subclass of Function): (NOTE: Don't provide this; already provided by `@classmethod`)
        Current function, such as Add, Sub, etc. args (tuple): arguments for the subclass's `.forward()`.
        (google "python asterisk arg")
        Returns:
            Tensor: Output tensor from operation that stores the current node.
        """
        # Creates BackwardFunction obj representing the current node
        backward_function = BackwardFunction(cls)
        # Run subclass's forward with context manager and operation input args
        output_tensor = cls.forward(backward_function.ctx, *args)
        # For each parent tensor in args, add their node to `backward_function.next_functions`
        isConstant = 0
        for parent in args:
            if type(parent) == tensor.Tensor:
                if parent.is_leaf and parent.requires_grad:
                    # accumulate the gradient so we initialize accumulate grad
                    parent.gard_fn = parent.acc()
                    next_function = parent.gard_fn
                    # then check if this is a backward function node
                elif not parent.is_leaf and parent.requires_grad:
                    # backward function
                    next_function = parent.grad_fn
                    # constant node
                else:
                    isConstant = 1
                    next_function = None
                backward_function.next_functions.append(next_function)
            # Store current node in output tensor 
        # output_tensor.grad_fn = backward_function
        if(isConstant==0):
            output_tensor.grad_fn = backward_function
            output_tensor.requires_grad = True

        return output_tensor


class ContextManager:
    """Used to pass variables between a function's `.forward()` and `.backward()`.
    (Argument "ctx" in these functions)

    To store a tensor:
    >>> ctx.save_for_backward(<tensors>, <to>, <store>)

    To store other variables (like integers):
    >>> ctx.<some_name> = <some_variable>
    """

    def __init__(self):
        self.saved_tensors = []  # list that TENSORS get stored in


    def save_for_backward(self, *args):
        """Saves TENSORS only
        See example above for storing other data types.
        Args:
            args (Tensor(s)): Tensors to store
        """
        for arg in args:
            # Raises error if arg is not tensor (i warned you)
            if type(arg).__name__ != "Tensor":
                raise Exception(
                    "Got type {} of object {}. \nOnly Tensors should be saved in save_for_backward. For saving constants, just save directly as a new attribute.".format(
                        type(arg), arg
                    )
                )

            self.saved_tensors.append(arg.copy())


class BackwardFunction:
    """Representing an intermediate node where gradient must be passed.
    Stored on output tensor of operation during `Function.apply()`

    Args:
        cls (subclass of Function): Operation being run. Don't worry about this;
                                    already handled in `Function.apply()`
    """
    def __init__(self, cls):
        self.ctx = ContextManager()  # Just in case args need to be passed (see above)
        self._forward_cls = cls

        # Nodes of parents, populated in `Function.apply`
        self.next_functions = []

        # The name of the operation as a string (for convenience)
        self.function_name = cls.__name__

    def apply(self, *args):
        """Generates gradient by running the operation's `.backward()`.
        Args:
            args: Args for the operation's `.backward()`
        Returns:
            Tensor: gradient of parent's output w.r.t. current output
        """
        # Note that we've already provided the ContextManager
        return self._forward_cls.backward(self.ctx, *args)
