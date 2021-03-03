import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from Encoder import Encoder
from Decoder import Decoder

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(0.5, 0.5),
])
train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1000, shuffle=True, num_workers=0, pin_memory=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0
)

device_string = "cuda" if torch.cuda.is_available() else "cpu"
print("using " + device_string)
device = torch.device(device_string)
encoder = Encoder().to(device)
encode_epochs = 50
for layer in range(2):
    if layer == 0:
        encoder.hebb1.training = True
        encoder.hebb2.training = False
    elif layer == 1:
        encoder.hebb1.training = False
        encoder.hebb2.training = True
    for epoch in range(encode_epochs):
        batch = 1
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)
            output = encoder(batch_features)
            print("Epoch: {}/{}, Batch: {}/{}".format(epoch + 1, encode_epochs, batch, len(train_loader)))
            batch += 1
encoder.toggle_training(False)

decoder = Decoder().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
decode_epochs = 50
for epoch in range(decode_epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # get code from encoder
        with torch.no_grad():
            code = encoder(batch_features)
        # compute reconstructions
        outputs = decoder(code)
        # compute training reconstruction loss
        train_loss = loss_fn(outputs, batch_features)
        # compute accumulated gradients
        train_loss.backward()
        # perform parameter update based on current gradients
        optimizer.step()
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)
    # display the epoch training loss
    print("Epoch: {}/{}, Loss = {:.6f}".format(epoch + 1, decode_epochs, loss))

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_dataset.data[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        test_data = test_dataset.data[index]
        test_data = test_data.to(device)
        test_data = test_data.float()
        test_data = test_data.view(-1, 784)
        output = encoder(test_data)
        output = decoder(output)
        plt.imshow(output.cpu().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()