import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from Encoder import Encoder
from Decoder import Decoder
from AutoEncoder import AutoEncoder

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

encoder.toggle_training(True)
for epoch in range(1):
    batch = 1
    for batch_features, _ in train_loader:
        if batch > 1:
            continue
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        print(encoder(batch_features))
        print("encoded batch : {}/{}".format(batch, len(train_loader)))
        batch += 1
encoder.toggle_training(False)

# epochs = 1
# for epoch in range(epochs):
#     loss = 0
#     for batch_features, _ in train_loader:
#         # reshape mini-batch data to [N, 784] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 784).to(device)
#         # reset the gradients back to zero
#         # PyTorch accumulates gradients on subsequent backward passes
#         optimizer.zero_grad()
#         # get code from encoder
#         code = encoder(batch_features)
#         print(code)
#         # compute reconstructions
#         outputs = decoder(code)
#         # compute training reconstruction loss
#         train_loss = loss_fn(outputs, batch_features)
#         # compute accumulated gradients
#         train_loss.backward()
#         # perform parameter update based on current gradients
#         optimizer.step()
#         # add the mini-batch training loss to epoch loss
#         loss += train_loss.item()

#     # compute the epoch training loss
#     loss = loss / len(train_loader)
#     # display the epoch training loss
#     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

# with torch.no_grad():
#     number = 10
#     plt.figure(figsize=(20, 4))
#     for index in range(number):
#         # display original
#         ax = plt.subplot(2, number, index + 1)
#         plt.imshow(test_dataset.data[index].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         # display reconstruction
#         ax = plt.subplot(2, number, index + 1 + number)
#         test_data = test_dataset.data[index]
#         test_data = test_data.to(device)
#         test_data = test_data.float()
#         test_data = test_data.view(-1, 784)
#         output = encoder(test_data)
#         output = decoder(output)
#         plt.imshow(output.cpu().reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()