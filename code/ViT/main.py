import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


DATA_DIR = "/tmp/MNIST"
BATCH_SIZE = 32


# class ViT():
#     def __init__(self, N, D, H) -> None:
#         self.flatten = nn.Flatten()
#         self.linear_projection = nn.Linear(N, D)
#         self.msa1 = torch.nn.MultiheadAttention(
#             embed_dim=D, num_heads=H, dropout=True
#         )
#         self.norm1_1 = nn.LayerNorm()
#         self.norm1_2 = nn.LayerNorm()
#         self.norm1_3 = nn.LayerNorm()
#         self.mlp1_1 = nn.Linear(D, D)
#         self.mlp1_2 = nn.Linear(D, D)

#         self.encoders = []
#         for i in range(16):
#             self.encoders.append(
#                 [
#                     nn.LayerNorm(),
#                     nn.MultiheadAttention(
#                         embed_dim=D,
#                         num_heads=H,
#                         dropout=True
#                     ),
#                     nn.LayerNorm(),
#                     nn.Linear(D, D),
#                     nn.GELU(),
#                     nn.LayerNorm(),
#                     nn.Linear(D, D),
#                     nn.LayerNorm(),
#                     nn.GELU(),
#                     nn.LayerNorm()
#                 ]
#             )
#         return

#     def forward(self, x: torch.Tensor, y, training=False):
#         x = self.flatten(x)
#         x = self.linear_projection(x)
#         z = [y, *x]
#         z = self.norm1(z)
#         z = self.msa(z)
#         z = self.norm2(z)
#         z = self.mlp1(z)
#         z = F.gelu(z)
#         z = self.mlp2(z)
#         z = F.gelu(z)
#         z = self.norm3(z)
#         return z[0]

class ViT(torch.nn.Module):
    def __init__(self, P, N, D, H, MLP) -> None:
        super(ViT, self).__init__()
        self.D = D
        self.N = N
        self.location_encoding = torch.Tensor(
            [[i/(N + 1) for i in range(0, 17)]])
        self.flatten = nn.Flatten(2)
        self.class_projection = nn.Linear(D, D)
        self.location_projection = nn.Linear(1, D)  # N + 1, D)
        self.patch_projection = nn.Linear(P*P, D)
        self.MSA1 = nn.MultiheadAttention(D, H)
        self.MLP1 = nn.Linear(D, D)
        self.MSA2 = nn.MultiheadAttention(D, H)
        self.MLP2 = nn.Linear(D, D)
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, H, 2500, activation="gelu"),
            num_layers=3
        )
        # self.encoder1 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         D, H, MLP, activation='gelu'
        #     ),
        #     num_layers=2
        # )
        self.mlp1 = nn.Linear(self.D, 250)
        self.mlp2 = nn.Linear(250, 100)
        self.mlp3 = nn.Linear(100, 10)
        self.amax = torch.argmax

    def forward(self, x):
        class_random = torch.normal(0, 1, size=(x.shape[0], 1, self.D))
        patch_location_encoding = torch.cat(
            [self.location_encoding]*x.shape[0]
        ).unsqueeze(-1)
        x_loc = self.location_projection(patch_location_encoding)
        x = self.flatten(x)
        x_p = self.patch_projection(x)
        encoding = torch.cat([class_random, x_p], dim=1) + x_loc
        encoding = F.normalize(encoding)
        z = self.encoder1(encoding)
        head_code = z[:, 0]
        # x = F.relu(head_code)
        # x = self.mlp1(x)
        # x = F.relu(x)
        # x = self.mlp2(x)
        # x = F.tanh(x)
        # x = self.mlp3(x)
        x = self.mlp1(head_code)
        x = self.mlp2(x)
        x = F.tanh(x)
        x = self.mlp3(x)
        output = F.log_softmax(x, dim=1)
        return output

        # z = F.layer_norm(encoding, normalized_shape=(
        #     x.shape[0], self.N + 1, self.D))
        # z = self.MSA1(z) + z
        # z = F.layer_norm(z)
        # print(z.shape)
        # z = self.MLP1(z) + z
        # z = F.layer_norm(z)
        # z = self.MSA2(z) + z
        # z = F.layer_norm(z)
        # z = self.MLP2(z) + z
        # self.amax(z)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        print(output[0])
        return output


def load_data_cnn():
    trainset = torchvision.datasets.MNIST(DATA_DIR, True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = torchvision.datasets.MNIST(DATA_DIR, False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True
    )
    return trainloader, testloader


def load_data_ViT():
    trainset = torchvision.datasets.MNIST(DATA_DIR, True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        cut_image
    ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True)
    testset = torchvision.datasets.MNIST(DATA_DIR, False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        cut_image
    ]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True
    )
    return trainloader, testloader


def cut_image(image):
    image = image.squeeze()
    cuts = [torch.hsplit(img, 4) for img in torch.vsplit(image, 4)]
    DD = []
    for LL in cuts:
        for i in LL:
            DD.append(i.unsqueeze(0))
    DD = np.concatenate(DD)
    return DD


batch_size_train = 64
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1


def train(network, optimizer, train_loader, epoch):
    network.train()
    epoch_train_losses = []
    epoch_train_counter = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            epoch_train_losses.append(loss.item())
            epoch_train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')
    return epoch_train_losses, epoch_train_counter


def test(network, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    epoc_loss = test_loss
    epoc_counter = len(train_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses, test_counter


# Training CNN

# train_loader, test_loader = load_data_cnn()
# n_epochs = 1
# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = []
# network = CNN()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                       momentum=momentum)
# for epoch in range(1, n_epochs + 1):
#     epoch_losses, epoch_counter = train(
#         network, optimizer, train_loader, epoch)
#     epoch_loss, epoch_count = test(network, test_loader)
#     train_losses += epoch_losses
#     train_counter += epoch_counter
#     test_losses.append(epoch_loss)
#     test_counter.append(epoch_count)

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# # plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig.savefig("CNNLoss")


# Training ViT
train_loader, test_loader = load_data_ViT()
n_epochs = 10
train_losses = []
train_counter = []
test_losses = []
test_counter = []
# P, N, D, H, MLP
network = ViT(7, 16, 360, 12, 10)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
for epoch in range(1, n_epochs + 1):
    epoch_losses, epoch_counter = train(
        network, optimizer, train_loader, epoch)
    epoch_loss, epoch_count = test(network, test_loader)
    train_losses += epoch_losses
    train_counter += epoch_counter
    test_losses.append(epoch_loss)
    test_counter.append(epoch_count)
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.savefig("ViTLoss")
