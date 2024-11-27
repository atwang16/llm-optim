import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    # transform to normalize data and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((14, 14)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Leave only 0 and 1 digits
    train_dataset.data = train_dataset.data[(train_dataset.targets == 0) | (train_dataset.targets == 1)]
    train_dataset.targets = train_dataset.targets[(train_dataset.targets == 0) | (train_dataset.targets == 1)]
    test_dataset.data = test_dataset.data[(test_dataset.targets == 0) | (test_dataset.targets == 1)]
    test_dataset.targets = test_dataset.targets[(test_dataset.targets == 0) | (test_dataset.targets == 1)]

    # dataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self, input_size=14*14, num_classes=2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)  # flatten 14*14 images to 196 vector
        x = self.fc1(x)      # output layer
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, output_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()              # clear gradients from the previous step
            output = model(data)               # forward pass
            loss = criterion(output, target)
            loss.backward()                    # backward pass (gradient computation)
            optimizer.step()                   # update weights

            total_loss += loss.item()
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss/len(train_loader),
        }
        torch.save(checkpoint, f"{output_dir}/ckpt_{epoch:03d}.pth")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")


def evaluate_model(model, test_loader):
    model.eval()  # evaluation mode
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
    batch_size = 12665
    learning_rate = 0.01
    num_epochs = 50
    hidden_size = 128
    output_dir = "../toy_mnist_mlp_ckpt_lr_0.01"

    train_loader, test_loader = get_data_loaders(batch_size)
    # initialize model, loss, and optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("training the MLP model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, output_dir)

    print("evaluating the model...")
    evaluate_model(model, test_loader)
