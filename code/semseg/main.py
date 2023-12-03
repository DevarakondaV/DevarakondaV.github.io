# Exploring semantic segmentation models.
# Code for MRCNN model
import os
import json
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from unet import UNet
from PIL import Image


def load_data(train_images_dir, test_images_dir):
    thread_count = 5
    coco_train_data = torchvision.datasets.CocoDetection(
        train_images_dir,
        '/home/vishnu/Downloads/cocoVal/instances_val2017.json',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    coco_test_data = torchvision.datasets.CocoDetection(
        test_images_dir,
        '/home/vishnu/Downloads/cocoVal/instances_val2017.json',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        coco_train_data,
        batch_size=4,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        coco_test_data,
        batch_size=4,
        shuffle=True
    )
    return train_loader, test_loader

# model = torchvision.models.detection.maskrcnn_resnet50_fpn(progress=True)


log_interval = 10
DEVICE = "cuda"


def train(network, optimizer, train_loader, loss_function, DEVICE, epoch):
    log_interval = 10
    network.train()
    epoch_train_losses = []
    epoch_train_counter = []
    batch_idx = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_function(output, target)
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
        batch_idx += 1
    return epoch_train_losses, epoch_train_counter


def test(network, loss_function, test_loader):
    network.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = network(data)
            test_loss += loss_function(output, target).item()
    test_loss /= len(test_loader.dataset)
    epoc_loss = test_loss
    # epoc_counter = len(train_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))
    # return test_losses, test_counter


def move_images(all_images_dir, train_images_dir, test_images_dir):
    split = 0.8
    all_images = os.listdir(all_images_dir)
    size = len(all_images)
    train_images_len = int(size * split)
    train_images = all_images[:train_images_len]
    test_images = all_images[train_images_len:]
    train_images = [all_images_dir + file for file in train_images]
    test_images = [all_images_dir + file for file in test_images]

    for file in train_images:
        file_name = file.split("/")[-1]
        os.rename(file, train_images_dir + file_name)

    for file in test_images:
        file_name = file.split("/")[-1]
        os.rename(file, test_images_dir + file_name)
    # test_images = []


def check_file_size(dir):
    for file in os.listdir(dir):
        x = np.array(Image.open(dir + file).pad((572, 572, 3)))
        print(x.shape)


if __name__ == "__main__":
    all_images_dir = "/home/vishnu/Downloads/cocoVal/allImages/"
    train_images_dir = "/home/vishnu/Downloads/cocoVal/train/"
    test_images_dir = "/home/vishnu/Downloads/cocoVal/test/"
    ann_file = '/home/vishnu/Downloads/cocoVal/instances_val2017.json'

    # check_file_size(test_images_dir)

    train_data, test_data = load_data(train_images_dir, test_images_dir)
    n_epochs = 1
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    network = UNet()

    for x, y in train_data:
        print(x.shape, y.shape)
        exit()

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
    # # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # # fig.savefig("CNNLoss")
