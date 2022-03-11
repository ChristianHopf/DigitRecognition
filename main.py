import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 10
classes = 10
learning_rate = 0.25
batch_size = 50
input_size = 784
hidden_layer = 100

train_data = torchvision.datasets.MNIST(root = "./dataset", train = True, transform = transforms.ToTensor(), download = True)
test_data = torchvision.datasets.MNIST(root = "./dataset", train = False, transform = transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle = False, num_workers = 2)

class DigitRecognizer(nn.Module):

    def __init__(self, input_size, hidden_layer, classes):
        super(DigitRecognizer, self).__init__()
        self.input = nn.Linear(in_features = input_size, out_features = hidden_layer)
        self.relu1 = nn.ReLU()
        self.hidden = nn.Linear(in_features = hidden_layer, out_features = hidden_layer)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(in_features = hidden_layer, out_features = classes)

    def forward(self, X):
        model = self.input(X)
        model = self.relu1(model)
        model = self.hidden(model)
        model = self.relu2(model)
        model = self.output(model)

        return model

model = DigitRecognizer(input_size, hidden_layer, classes)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

samples = len(train_loader)
print(samples)

model = model.to(device)

for epoch in range(epochs):
    for step, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, step: {}/{}, loss: {:.4f}".format(epoch, epochs, step, samples, loss.item()))