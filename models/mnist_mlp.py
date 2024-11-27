import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    # transform to normalize data and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # mean=0.5, std=0.5
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # dataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)         # flatten 28x28 images to 784 vector
        x = self.relu(self.fc1(x)) # first layer + ReLU activation (nonlinear)
        x = self.fc2(x)            # output layer
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()  # evaluation mode ??
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient calculation for evaluation ??
        for data, target in test_loader:
            output = model(data)                      # forward pass
            pred = output.argmax(dim=1)              # get predicted class
            correct += (pred == target).sum().item() # count correct predictions
            total += target.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10
    hidden_size = 128

    train_loader, test_loader = get_data_loaders(batch_size)

    # initialize model, loss, and optimizer
    model = MLP(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()  # Logistic loss ??
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("training the MLP model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    print("evaluating the model...")
    evaluate_model(model, test_loader)
