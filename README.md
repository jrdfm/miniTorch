# miniTorch

Automatic differentiation engine that is similar to PyTorch, used to automatically calculate the derivative
/ gradient of any computable function.
Inspired by Karphaty's [micrograd](https://github.com/karpathy/micrograd), geohot's [tinygrad](https://github.com/geohot/tinygrad) and [nanograd](https://github.com/PABannier/nanograd) and CMU 11-785 Introduction to Deep Learning notes.

# Features

- PyTorch-like backpropagation, reverse autodifferentiation,  on dynamically built computational graph, DAG.
- Activations: ReLU, Sigmoid, tanh, Swish, ELU, LeakyReLU
- Convolutions: Conv1d, Conv2d, MaxPool2d, AvgPool2d
- Layers: Linear, BatchNorm1d, BatchNorm2d, Flatten, Dropout
- Optimizers: SGD, Adam
- Loss: CrossEntropyLoss, Mean squared error
- RNN, GRU


# Todo
- Computational graph visualizer (see example)
- Weight initialization: Glorot uniform, Glorot normal, Kaiming uniform, Kaiming normal
- Activations: Swish, ELU, LeakyReLU
- Optimizers: AdamW