import torch

from datasets import mnist

from TheBrain import TheBrain

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_loader, test_loader = mnist.get_loaders(train_bz=128, test_bz=1000, normalize_0_1=True)
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
print("using " + device_string)

model = TheBrain(input_size=784, output_size=10, device=device)

model.train_first_layer(n_epochs=200, train_loader=train_loader)
model.train_second_layer(n_epochs=200, train_loader=train_loader)
model.train_output_layer(n_epochs=200, train_loader=train_loader, test_loader=test_loader)
