import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x  # Save input for residual connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Add residual connection
        return F.relu(out)


class Net(nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        # Initial CNN layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ResNet Block
        self.resnet_block = nn.Sequential(
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32),
            ResNetBlock(32, 32)
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)  # Output 10 classes

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Initial CNN layer with batch norm
        x = self.resnet_block(x)  # ResNet block
        x = self.global_avg_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Fully connected layer 2
        x = self.softmax(x)  # Softmax for classification
        return x



def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set"""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


if __name__ == "__main__":
    model = Net(10)
    print(model)

    # Dummy input for testing (batch size = 1, channels = 1, height = 64, width = 64)
    dummy_input = torch.randn(1, 1, 11, 11)
    output = model(dummy_input)
    print(output)
