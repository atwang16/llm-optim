"""
Linear Perceptron for Full MNIST Classification
===============================================

This script implements a linear perceptron for the MNIST dataset.
The model consists of a single fully connected layer with no hidden layers and no non-linearity.

Features:
- Full MNIST classification (10 classes).
- Minimal architecture with 7850 total parameters.
- Training and evaluation pipelines with model checkpointing.

Author: Amirhossein Etaati
Date: 2024-12-02
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    """
    Prepare DataLoaders for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: DataLoader for training and testing datasets.
    """
    # transform: normalize and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts image to a tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # standardize with MNIST mean and std
    ])

    # load the full MNIST dataset (all 10 digits - no filtering of digits)
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class LinearPerceptron(nn.Module):
    """
    Linear Perceptron Model for MNIST Classification.

    Attributes:
        flatten (nn.Flatten): Flattens input images to vectors.
        fc1 (nn.Linear): Fully connected layer (784 -> 10).
    """
    def __init__(self, input_size=28*28, num_classes=10):
        super(LinearPerceptron, self).__init__()
        self.flatten = nn.Flatten() # to flatten the 28x28 images to 784-vectors to make it compatible with the fully connected (linear) layers
        self.fc1 = nn.Linear(input_size, num_classes) # fully connected layer
    
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits for each class.
        """
        x = self.flatten(x)  # flatten input: 28*28 images to 784-vectors
        x = self.fc1(x)      # linear transformation: output = W . input + b
        return x


def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=10, output_dir="checkpoints"):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of training epochs.
        output_dir (str): Directory to save model checkpoints.
    """

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {output_dir}: {e}")
        exit(1)

    model.train()  # set model to training mode
    accs = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            optimizer.zero_grad()              # clear gradients from the previous step
            output = model(data)               # forward pass
            loss = criterion(output, target)   # compute loss
            loss.backward()                    # backward pass (gradient computation)
            optimizer.step()                   # update weights
            total_loss += loss.item()   

        # save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss/len(train_loader),
        }
        with torch.no_grad():
            acc = evaluate_model(model, test_loader)
            accs.append(acc)
        torch.save(checkpoint, f"{output_dir}/ckpt_{(epoch+1):03d}.pth")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    from matplotlib import pyplot as plt
    plt.plot(accs)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over steps")
    plt.savefig(f"{output_dir}/accuracy.png")
    plt.close()


def evaluate_model(model, test_loader):
    """
    Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for test data.

    Returns:
        float: Test accuracy as a percentage.
    """
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient calculation for evaluation
        for data, target in test_loader:
            output = model(data)                      # forward pass
            pred = output.argmax(dim=1)               # get predicted class
            correct += (pred == target).sum().item()  # count correct predictions
            total += target.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # hyperparameters
    batch_size = 1000  # why?? isn't it large?
    learning_rate = 0.001
    num_epochs = 50
    output_dir = os.path.abspath("../mnist_bsizedsize_1e-3_50_batches/ckpts")

    train_loader, test_loader = get_data_loaders(batch_size)

    # initialize model, loss, and optimizer
    model = LinearPerceptron().to("cuda")
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("Training the LinearPerceptron model...")
    train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs, output_dir)

    print("Evaluating the model...")
    evaluate_model(model, test_loader)
    evaluate_model(model, test_loader)
