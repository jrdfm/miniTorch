"""Problem 3 - Training on MNIST"""
import numpy as np

from mytorch.optim.sgd import SGD
from mytorch.optim.adam import Adam
from mytorch.nn.activations import ReLU
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.linear import Linear
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.sequential import Sequential
from mytorch.tensor import Tensor
from tqdm import trange, tqdm
# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion

    model = Sequential(Linear(784, 20),BatchNorm1d(20),ReLU(),Linear(20,10))
    # model = Sequential(Linear(784, 100),BatchNorm1d(100),ReLU(),Linear(100,20),BatchNorm1d(20),ReLU(),Linear(20, 10))
    # Accuracies [0.9786, 0.9772, 0.9766, 0.9782, 0.9792, 0.9792]

    optimizer = SGD(model.parameters(), lr = 0.1, momentum= 0.1)
    #optimizer = Adam(model.parameters())

    criterion = CrossEntropyLoss()



    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=5)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=5):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    print("Init Train")
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss, total_acc = 0, 0
        val_accuracies = []
        train_size = train_x.shape[0]
        
        SH = False
        if SH:
            ix = np.arange(train_size)
            np.random.shuffle(ix)
            train_x = train_x[ix]
            train_y = train_y[ix]

        batches = [(x, y) for x, y in zip(np.vsplit(train_x, train_size / BATCH_SIZE),
                                          np.split(train_y, train_size / BATCH_SIZE))]

        for i, (batch_data, batch_labels) in enumerate(batches):
            optimizer.zero_grad()
            out = model.forward(Tensor(batch_data))
            loss = criterion(out, Tensor(batch_labels))
            total_loss += loss.data
            
            acc = (np.argmax(out.data, axis=1) == batch_labels.data).sum() / batch_labels.shape[0]
            total_acc += acc

            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()
        print(f'Epoch {epoch+1}, Loss: {total_loss/(i+1)}, Accuracy: {total_acc/(i+1)}')

    return val_accuracies

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    model.eval()
    val_size = val_x.shape[0]
    batches = [(x, y) for x, y in
               zip(np.vsplit(val_x, val_size / BATCH_SIZE), np.split(val_y, val_size / BATCH_SIZE))]
    num_correct = 0

    for i, (batch_data, batch_labels) in enumerate(batches):
        out = model.forward(Tensor(batch_data))
        batch_preds = np.argmax(out.data,axis=1)
        num_correct += batch_preds == batch_labels

    accuracy = num_correct.sum() / len(val_x)

    return accuracy
