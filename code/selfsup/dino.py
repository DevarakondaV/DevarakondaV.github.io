import os
import logging
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import datetime
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Attention(torch.nn.Module):

    def __init__(
        self,
        dimensions,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0
    ) -> None:
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dimensions = dimensions
        head_dim = self.dimensions // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(
            self.dimensions,
            self.dimensions * 3,
            bias=qkv_bias
        )
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.dimensions, self.dimensions)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        return

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # torch.Size([1, 1, 3, 10, 50])
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
        #                           self.num_heads)
        # q, k, v = qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(torch.nn.Module):

    def __init__(
        self,
        dimensions,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0
    ) -> None:
        super(Block, self).__init__()
        self.attn = Attention(dimensions, num_heads,
                              qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = torch.nn.Identity()
        self.norm1 = torch.nn.LayerNorm(dimensions)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 4 * dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(4 * dimensions, dimensions),
            torch.nn.Dropout(drop)
        )
        self.norm2 = torch.nn.LayerNorm(dimensions)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.MLP(self.norm2(x)))
        return x


class ViT(torch.nn.Module):

    def __init__(self, dimensions, num_heads, num_classes, num_positions, patch_size) -> None:
        super(ViT, self).__init__()
        self.class_random = torch.normal(0, 1, size=(1, 1, dimensions))
        self.location_encoding = torch.tensor(
            [[i for i in range(0, 17)]], dtype=torch.int32)
        self.position_embeddings = torch.nn.Embedding(
            num_positions + 1, dimensions
        )
        self.patch_projection = torch.nn.Linear(
            patch_size*patch_size, dimensions
        )
        self.flatten = torch.nn.Flatten(2)
        self.norm = torch.nn.LayerNorm(dimensions)
        self.block1 = Block(dimensions, num_heads)
        self.block2 = Block(dimensions, num_heads)
        self.block3 = Block(dimensions, num_heads)
        self.block4 = Block(dimensions, num_heads)
        self.norm = torch.nn.LayerNorm(dimensions)
        return

    def forward(self, x):
        x = torch.from_numpy(x)
        class_random = self.class_random.repeat(x.shape[0], 1, 1)
        x_loc = self.position_embeddings(self.location_encoding)
        x = self.flatten(x)
        x_p = self.patch_projection(x)
        encoding = torch.cat([class_random, x_p], dim=1) + x_loc
        encoding = self.norm(encoding)
        x = self.block1(encoding)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_attention(self):
        class_random = self.class_random.repeat(x.shape[0], 1, 1)
        x_loc = self.position_embeddings(self.location_encoding)
        x = self.flatten(x)
        x_p = self.patch_projection(x)
        encoding = torch.cat([class_random, x_p], dim=1) + x_loc
        encoding = self.norm(encoding)
        x = self.block1(encoding)
        x = self.block2(x)
        x = self.block3(x)
        attn = self.block4(x, return_attention=True)
        return attn


def loss_fn(student_outputs, t1, t2, dino_train_params):
    t1 = t1.detach()
    t2 = t2.detach()
    temp_teacher = dino_train_params["temp_teacher"]
    temp_student = dino_train_params["temp_student"]
    center = dino_train_params["C"]
    t1 = t1.unsqueeze(1)
    t2 = t2.unsqueeze(1)
    t1 = t1.repeat(1, student_outputs.shape[1], 1)
    t2 = t2.repeat(1, student_outputs.shape[1], 1)
    t1 = F.softmax((t1 - center) / temp_teacher, dim=1)
    t2 = F.softmax((t2 - center) / temp_teacher, dim=1)
    s = F.softmax(student_outputs / temp_student, dim=1)
    loss1 = - (t1 * torch.log(s)).sum(dim=1).mean()
    loss2 = - (t2 * torch.log(s)).sum(dim=1).mean()
    return (loss1 / 2) + (loss2 / 2)


class DINO(torch.nn.Module):

    def __init__(self) -> None:
        super(DINO, self).__init__()
        # self.teacher = ViT(7, 16, 500, 10, 10)
        # self.student = ViT(7, 16, 500, 10, 10)
        self.teacher = ViT(500, 10, 10, 16, 7)
        self.student = ViT(500, 10, 10, 16, 7)
        self.param_map = {}
        for sparam, tparam in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            tparam = tparam[0]
            sparam = sparam[0]
            self.param_map[tparam] = sparam
        self.global_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(24, 24)),
            torchvision.transforms.Resize((28, 28))
        ])
        self.local_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((12, 12)),
            torchvision.transforms.Resize((28, 28))
        ])

    def global_augment(self, x):
        x = self.global_transforms(x)
        return cut_image(x)

    def local_augment(self, x, crop_count=5):
        # x = [cut_image(self.local_transforms(x)) for i in range(crop_count)]
        # x = torch.concatenate(x, dim=1)
        return cut_image(self.local_transforms(x))
        # return x

    def forward(self, x):
        x1g, x2g = self.global_augment(x), self.global_augment(x)
        xl = self.local_augment(x)
        s1 = self.student(x1g)
        s2 = self.student(x2g)
        s3 = self.student(xl)
        # s3 = [self.student(xl[:, i, :, :].unsqueeze(1)).unsqueeze(1)
        #       for i in range(xl.shape[1])]
        with torch.no_grad():
            t1 = self.teacher(x1g)
            t2 = self.teacher(x2g)
        s1 = s1.unsqueeze(1)
        s2 = s2.unsqueeze(1)
        s3 = s3.unsqueeze(1)
        student_outputs = torch.concatenate(
            [s1, s2, s3], dim=1)  # + s3, dim=1)
        return student_outputs, t1, t2


def train_dino():
    DIR = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/tmp"
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    CHECKPOINT_DIR = f"results/checkpoints"
    EPOCHS = 1
    BATCH_SIZE = 16
    LEARNING_RATE = 1E-3
    DEVICE = 'cpu'
    CHECKPOINT = "NEW"
    OPTIMIZER = "Adam"
    SAVE_CHECKPOINT = True
    train_timestamp = datetime.datetime.now().isoformat().replace(":", ".")
    TRAINING_DIR = f"{DIR}/training/DINO_{train_timestamp}"
    TENSORBOARD_DIR = f"{TRAINING_DIR}/tensorboard/DINO_{train_timestamp}"
    writer = SummaryWriter(TENSORBOARD_DIR)

    dataset = torchvision.datasets.MNIST(
        "/tmp", True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ]))

    loader_args = dict(batch_size=BATCH_SIZE,
                       num_workers=os.cpu_count(), pin_memory=True)
    dataset = DataLoader(dataset, shuffle=True, **loader_args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'''Starting training:
        Epochs:          {EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Checkpoints:     {CHECKPOINT}
        Device:          {DEVICE}
        Optimizer:       {OPTIMIZER}
    ''')

    dino = DINO()
    dino.student.to(DEVICE)
    dino.teacher.to(DEVICE)
    # dino.student.type(torch.bfloat16)
    # dino.teacher.type(torch.bfloat16)
    optimizer = torch.optim.Adam(dino.student.parameters(), lr=1E-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    dino_train_params = {
        "temp_teacher": 0.05,
        "temp_student": 0.1,
        "C": 0,
        "l": 0.998,
        "m": 0.998
    }
    global_step = 0
    for epoch in tqdm(range(EPOCHS), desc="Running Epoch", unit="epoch"):
        dino.student.train()
        epoch_loss = 0
        total_count = 0
        for image_batch, labels in tqdm(dataset, desc="Running Batch", unit="Batch"):
            image_batch = image_batch.to(DEVICE)  # .type(torch.bfloat16)
            student_outputs, t1, t2 = dino(image_batch)
            loss = loss_fn(student_outputs, t1, t2, dino_train_params)
            loss.backward()
            optimizer.step()
            l = dino_train_params["l"]
            m = dino_train_params["m"]
            C = dino_train_params["C"]
            for param in dino.teacher.named_parameters():
                param_name = param[0]
                param = l * param[1] + (l - 1) * \
                    dino.student.get_parameter(dino.param_map[param_name])
            C = m * C + (1-m)*torch.concat([t1, t2]).mean(dim=0)
            dino_train_params["C"] = C
            epoch_loss += loss.item()
            writer.add_scalar(
                "Loss", loss.item(), total_count
            )
            print("Loss: ", loss.item())
            total_count += 1
        scheduler.step(epoch_loss / total_count)
        global_step += 1
        if SAVE_CHECKPOINT:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict = dino.student.state_dict()
            torch.save(state_dict, "{}/checkpoint_epoch{}.pth".format(
                CHECKPOINT_DIR, epoch))
            logging.info(f'Checkpoint {epoch} saved!')
    return


def visualize_dino():
    CHECKPOINT_FILE = f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/results/checkpoints/checkpoint_epoch0.pth"
    DEVICE = 'cpu'

    dataset = torchvision.datasets.MNIST(
        "/tmp", True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)
            )
        ])
    )

    loader_args = dict(batch_size=1,
                       num_workers=os.cpu_count(), pin_memory=True)
    dataset = DataLoader(dataset, shuffle=True, **loader_args)

    dino = DINO()
    dino.student.to(DEVICE)
    dino.student.load_state_dict(torch.load(CHECKPOINT_FILE))
    dino.student.eval()
    print(dino)
    print(dino.student.encoder1.layers[-1].self_attn)
    for image_batch, labels in dataset:
        print("Label: ", labels, image_batch.shape)
        cv2.imwrite(
            "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/tmp/img.jpg",
            (image_batch.squeeze() * 255).detach().numpy().astype(np.uint8))
        image_item = F.interpolate(image_batch, size=(7, 7), mode="area")
        out = dino.student.retrieve_attention_heads(image_item)
        # out = dino.student(image_item)
        # heads = dino.student.heads
        # head_code = heads[:, 0].detach().numpy().reshape(1, 20, 25).squeeze()
        # img_code = (cv2.resize(head_code, (28, 28)) * 255).astype(np.uint8)
        # img_code = cv2.applyColorMap(img_code, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(
        #     "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/tmp/pred.jpg", img_code)
        break
    return


def cut_image(image):
    image = image.squeeze()
    cuts = [torch.tensor_split(img, 4, 2)
            for img in torch.tensor_split(image, 4, 1)]
    DD = []
    for LL in cuts:
        for i in LL:
            DD.append(i.unsqueeze(0))
    DD = np.concatenate(DD)
    return DD


def load_data_ViT():
    trainset = torchvision.datasets.MNIST("/tmp", True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        cut_image
    ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True)
    testset = torchvision.datasets.MNIST("/tmp", False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        cut_image
    ]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=True
    )
    return trainloader, testloader


if __name__ == "__main__":
    train_dino()
    # visualize_dino()

    # attn = Attention(500, 10)
    # attn.type(torch.float32)

    # block = Block(500, 10)
    # vit = ViT(7, 16, 500, 10, 10)

    # vit = ViT(500, 10, 10, 16, 7)
    # x = np.random.rand(1, 16, 7, 7)
    # x = torch.from_numpy(x).type(torch.float32)
    # vit(x)
