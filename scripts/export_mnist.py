import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import struct
import os

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_and_export():
    print("[1/3] Downloading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[2/3] Training the model")
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("[3/3] Exporting weights to a binary file...")
    os.makedirs('../app', exist_ok=True)
    export_path = '../app/mnist_weights.bin'

    with open(export_path, 'wb') as f:
        w1 = model.fc1.weight.detach().numpy().T
        b1 = model.fc1.bias.detach().numpy()
        f.write(w1.tobytes())
        f.write(b1.tobytes())

        w2 = model.fc2.weight.detach().numpy().T
        b2 = model.fc2.bias.detach().numpy()
        f.write(w2.tobytes())
        f.write(b2.tobytes())

    print(f"Weights saved in: {export_path}")

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    img, label = testset[0]

    with open('../app/test_image.bin', 'wb') as f:
        f.write(img.numpy().tobytes())

    print(f"Test image (number {label}) saved as: ../app/test_image.bin")

if __name__ == '__main__':
    train_and_export()