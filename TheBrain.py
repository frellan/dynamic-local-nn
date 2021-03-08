import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from AlignmentLoss import AlignmentLoss

LOG_INTERVAL = 100

class TheBrain(nn.Module):

    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

    def fc1_forward(self, x):
        x = self.fc1(x)
        return F.relu(x)

    def fc2_forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)

    def train_first_layer(self, n_epochs, train_loader):
        self.fc1.requires_grad_(True)
        self.fc2.requires_grad_(False)
        self.output_layer.requires_grad_(False)

        learning_rate = 0.01
        momentum = 0.5
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.__train_layer(
            forward_fn=self.fc1_forward,
            loss_fn=self.__alignment_loss,
            n_epochs=n_epochs,
            train_loader=train_loader,
            optimizer=optimizer)

        self.fc1.requires_grad_(True)
        self.fc2.requires_grad_(True)
        self.output_layer.requires_grad_(True)

    def train_second_layer(self, n_epochs, train_loader):
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(True)
        self.output_layer.requires_grad_(False)

        learning_rate = 0.01
        momentum = 0.5
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        self.__train_layer(
            forward_fn=self.fc2_forward,
            loss_fn=self.__alignment_loss,
            n_epochs=n_epochs,
            train_loader=train_loader,
            optimizer=optimizer)

        self.fc1.requires_grad_(True)
        self.fc2.requires_grad_(True)
        self.output_layer.requires_grad_(True)
    
    def train_output_layer(self, n_epochs, train_loader, test_loader):
        learning_rate = 0.01
        momentum = 0.5
        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)
        self.output_layer.requires_grad_(True)
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        def __train_output_layer(epoch):
            self.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        def __test_output_layer():
            self.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = self(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        __test_output_layer()
        for epoch in range(1, n_epochs + 1):
            __train_output_layer(epoch)
            __test_output_layer()

        self.fc1.requires_grad_(True)
        self.fc2.requires_grad_(True)
        self.output_layer.requires_grad_(True)

    def __alignment_loss(self, output, target): 
        size = output.shape[0]
        k1 = torch.tril(output @ output.t(), diagonal = -1)
        k2 = torch.zeros((size, size))
        for i in range(size):
            for j in range(i):
                k2[i, j] = target[i] == target[j]
        return torch.sum(k1 * k2) / (torch.sum(k2) * torch.norm(k1))
            
    def __train_layer(self, forward_fn, loss_fn, n_epochs, train_loader, optimizer):
        for epoch in range(1, n_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = forward_fn(data)
                loss = -loss_fn(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % LOG_INTERVAL == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))