import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # Mean of CIFAR-10
                             (0.2023, 0.1994, 0.2010))  # Std of CIFAR-10
    ])

    # Download and load the training and test datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    transform=transform, download=True)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)  # (3,32,32) -> (8,32,32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (8,32,32) -> (8,16,16)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)  # (8,16,16) -> (16,16,16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (16,16,16) -> (16,8,8)

        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)  # (16*8*8) -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


def train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs=20, output_dir="checkpoints"):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {output_dir}: {e}")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  # set model to training mode
    accs = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()              # clear gradients from the previous step
            output = model(data)               # forward pass
            loss = criterion(output, target)   # compute loss
            loss.backward()                    # backward pass (gradient computation)
            optimizer.step()                   # update weights
            total_loss += loss.item()

        # Evaluate the model on the test set
        with torch.no_grad():
            acc = evaluate_model(model, test_loader)
            accs.append(acc)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }
        checkpoint_path = f"{output_dir}/ckpt_{(epoch+1):03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {acc:.2f}%")

    # Plot accuracy over epochs
    plt.plot(accs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy over Epochs")
    plt.savefig(f"{output_dir}/accuracy.png")
    plt.close()


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient calculation for evaluation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)                      # forward pass
            pred = output.argmax(dim=1)               # get predicted class
            correct += (pred == target).sum().item()  # count correct predictions
            total += target.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 50
    output_dir = os.path.abspath("./cifar10_lightweight_cnn/checkpoints")

    # Prepare data loaders
    train_loader, test_loader = get_data_loaders(batch_size)

    # Initialize model, loss, and optimizer
    model = CNN(num_classes=10).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("Training the CNN model...")
    train_model(model, train_loader, criterion, optimizer, test_loader, num_epochs, output_dir)

    print("Evaluating the model on the test set...")
    evaluate_model(model, test_loader)
