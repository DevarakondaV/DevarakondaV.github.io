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
from PIL import Image, ImageOps


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

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
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

    def __init__(self, dimensions, num_heads, num_positions, patch_size) -> None:
        super(ViT, self).__init__()
        self.class_random = torch.normal(0, 1, size=(1, 1, dimensions))
        # self.location_encoding = torch.tensor(
        #     [[i for i in range(0, 17)]], dtype=torch.int32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.location_encoding = torch.tensor(
            [[i for i in range(0, num_positions + 1)]], dtype=torch.int32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # self.location_encoding = torch.tensor(
        #     [[i for i in range(0, 170)]], dtype=torch.int32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 4 * dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4 * dimensions, dimensions),
            torch.nn.GELU(),
            torch.nn.Dropout(),
            torch.nn.Linear(dimensions, 10),
        )
        return

    def forward(self, x):
        class_random = self.class_random.repeat(x.shape[0], 1, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        head = x[:, 0]
        return self.MLP(head)

    def get_last_attention(self, x):
        class_random = self.class_random.repeat(x.shape[0], 1, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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


def loss_fn(student_outputs, t1, t2, temp_teacher, temp_student, center):
    # t1 = t1.detach()
    # t2 = t2.detach()
    
    # t1 = t1.unsqueeze(1)
    # t2 = t2.unsqueeze(1)
    # t1 = t1.repeat(1, student_outputs.shape[1], 1)
    # t2 = t2.repeat(1, student_outputs.shape[1], 1)
    # t1 = F.softmax((t1 - center) / temp_teacher, dim=2)
    # t2 = F.softmax((t2 - center) / temp_teacher, dim=2)
    # s = F.softmax(student_outputs / temp_student, dim=2)

    student_outputs = [item / temp_student for item in student_outputs]
    t1 = F.softmax((t1 - center) / temp_teacher, dim=-1)
    t2 = F.softmax((t2 - center) / temp_teacher, dim=-1)
    teachers = [t1, t2]
    total_loss = 0
    n_loss_terms = 0
    for q in teachers:
        for v in range(len(student_outputs)):
            loss = torch.sum(-q * F.log_softmax(student_outputs[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    return total_loss




class DINO(torch.nn.Module):

    def __init__(self) -> None:
        super(DINO, self).__init__()
        # self.teacher = ViT(7, 16, 500, 10, 10)
        # self.student = ViT(7, 16, 500, 10, 10)
        # self.teacher = ViT(300, 5, 16, 7)
        # self.student = ViT(300, 5, 16, 7)
        
        self.teacher = ViT(300, 10, 196, 2)
        self.student = ViT(300, 10, 196, 2)

        # self.teacher = ViT(300, 10, 169, 8)
        # self.student = ViT(300, 10, 169, 8)
        self.param_map = {}
        for sparam, tparam in zip(self.student.named_parameters(), self.teacher.named_parameters()):
            tparam = tparam[0]
            sparam = sparam[0]
            self.param_map[tparam] = sparam
        self.global_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(24, 24)),
            torchvision.transforms.Resize((28, 28))
            # torchvision.transforms.RandomCrop(size=(90, 90)),
            # torchvision.transforms.Resize((104, 104))
        ])
        self.local_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((12, 12)),
            torchvision.transforms.Resize((28, 28))
            # torchvision.transforms.RandomResizedCrop((40, 40)),
            # torchvision.transforms.Resize((104, 104))
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
        xl1 = self.local_augment(x)
        xl2 = self.local_augment(x)
        xl3 = self.local_augment(x)
        xl4 = self.local_augment(x)
        s3 = self.student(xl1)
        s4 = self.student(xl2)
        s5 = self.student(xl3)
        s6 = self.student(xl4)
        with torch.no_grad():
            t1 = self.teacher(x1g)
            t2 = self.teacher(x2g)
        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
        s3 = s3.unsqueeze(1)
        s4 = s4.unsqueeze(1)
        s5 = s5.unsqueeze(1)
        s6 = s6.unsqueeze(1)
        student_outputs = [s3, s4, s5, s6]
        return student_outputs, t1, t2

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value)* (1 + np.cos(np.pi * iters / len(iters)))
    """
    print(
        warmup_schedule, 
        schedule, 
        warmup_schedule.shape, 
        schedule.shape, 
        type(warmup_schedule), 
        type(schedule), 
        np.expand_dims(warmup_schedule, 0).shape,
        np.expand_dims(schedule, 0).shape
        )
    """
    schedule = np.concatenate([warmup_schedule, schedule], axis = 0)
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class DogsDataset(torch.utils.data.Dataset):
    def __init__(self, images_data_dir, masks_data_dir):
        self.images_data_dir=images_data_dir
        self.masks_data_dir=masks_data_dir
        self.image_files=os.listdir(self.images_data_dir)

    def __len__(self):
        return len(self.image_files)
    
    @staticmethod
    def preprocess(pil_image, is_mask):
        w, h = pil_image.size
        # newW, newH = 256, 256
        # newW, newH = 28, 28
        newW, newH = 104, 104
        pil_image = pil_image.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img = np.asarray(pil_image)
        if is_mask:
            return img / 255.0
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
            return img
        
    def __getitem__(self, index):
        image_file=self.image_files[index]
        image_file_path=f"{self.images_data_dir}/{image_file}"
        mask_file_path=f"{self.masks_data_dir}/{image_file}"
        image = Image.open(image_file_path)
        image = ImageOps.grayscale(image)
        mask = Image.open(mask_file_path)
        image = self.preprocess(image, False)
        mask = self.preprocess(mask, True)
        image=torch.from_numpy(image).type(torch.float32)
        mask=torch.from_numpy(mask).type(torch.float32)
        return {
            "image": image,
            "mask": mask
        }

def train_dino():
    DIR = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/tmp"
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    DATA_IMG_DIR=f"{DIR}/pets/dogsImages"
    DATA_MASK_DIR=f"{DIR}/pets/dogsMasks"
    CHECKPOINT_DIR = f"results/checkpoints"
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    DEVICE = 'cpu'
    CHECKPOINT = "NEW"
    OPTIMIZER = "Adam"
    SAVE_CHECKPOINT = True
    train_timestamp = datetime.datetime.now().isoformat().replace(":", ".")
    TRAINING_DIR = f"{DIR}/training/DINO_{train_timestamp}"
    TENSORBOARD_DIR = f"{TRAINING_DIR}/tensorboard/DINO_{train_timestamp}"
    writer = SummaryWriter(TENSORBOARD_DIR)

    
    dataset = torchvision.datasets.MNIST(
        "/tmp", 
        True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #    (0.1307,), (0.3081,))
        ]))
    

    # dataset = DogsDataset(DATA_IMG_DIR, DATA_MASK_DIR)

    """
    dataset = torchvision.datasets.CIFAR10(
        "/tmp",
        True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Grayscale(),
            # torchvision.transforms.Normalize(
            #    (0.1307,), (0.3081,))
        ])
    )
    """

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
    dino.to(DEVICE)
    dino.student.to(DEVICE)
    dino.teacher.to(DEVICE)
    # dino.student.type(torch.bfloat16)
    # dino.teacher.type(torch.bfloat16)
    optimizer = torch.optim.Adam(dino.student.parameters(), lr=1E-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, 'min', patience=5)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    # )

    dino_train_params = {
        #"temp_teacher": 0.02,
        "temp_teacher": np.concatenate((
            np.linspace(0.04, 0.07, 30),
            np.ones(EPOCHS - 30)*0.07
        )),
        "temp_student": 0.1,
        "C": torch.zeros(1, 10).to(DEVICE),
        "l": 0.998,
        "m": cosine_scheduler(
            0.996, 1, EPOCHS, len(dataset)
        ),
    }
    global_step = 0
    lr_schedule = cosine_scheduler(
            0.0005 * (BATCH_SIZE / 256), 
            1E-6, 
            EPOCHS, 
            len(dataset), 
            warmup_epochs=10
    )
    weight_schedule = cosine_scheduler(
            0.04, 0.4, EPOCHS, len(dataset)
    )
    print("LR LEN: ", len(lr_schedule), EPOCHS, len(dataset))
    for epoch in tqdm(range(EPOCHS), desc="Running Epoch", unit="epoch"):
        dino.student.train()
        epoch_loss = 0
        total_count = 0
        looper = tqdm(dataset, desc="Running Batch", unit="Batch")
        # for image_batch, labels in looper:
        for i, item in enumerate(looper):
            schedulers_idx = ((epoch) * len(dataset)) + i
            lr_rate = lr_schedule[schedulers_idx]
            weight = weight_schedule[schedulers_idx]
            l = dino_train_params["l"]
            m = dino_train_params["m"][schedulers_idx]
            C = dino_train_params["C"]
            temp_teacher = dino_train_params["temp_teacher"][epoch]
            temp_student = dino_train_params["temp_student"]
            image_batch, labels = item
        # for item in looper:
            # image_batch = item["image"]
            image_batch = image_batch.to(DEVICE)
            student_outputs, t1, t2 = dino(image_batch)
            loss = loss_fn(student_outputs, t1, t2, temp_teacher,  temp_student, C)
            loss.backward()
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_rate
                if i == 0:
                    param_group["weight_decay"] = weight
            
            optimizer.step()
            with torch.no_grad():
                for param_name, param in dino.teacher.named_parameters():
                    param.data.mul_(l).add_((1-l)*dino.student.get_parameter(dino.param_map[param_name]))
                    # param.data = l * param + (1-l) * \
                    #     dino.student.get_parameter(dino.param_map[param_name])
            C = m * C + (1-m)*torch.concat([t1, t2], axis=1).mean(dim=0)
            dino_train_params["C"] = C
            epoch_loss += loss.item()
            looper.set_postfix_str({"Loss": loss.item()})
            # writer.add_scalar(
            #     "Loss", loss.item(), total_count
            # )
            # print("Loss: ", loss.item())
            total_count += 1
        print(f"Epoch Loss: {epoch}: {epoch_loss/total_count}", epoch)
        writer.add_scalar(
                "Epoch Loss", epoch_loss/total_count, epoch
        )
        # scheduler.step(epoch_loss / total_count)
        global_step += 1
        if SAVE_CHECKPOINT:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict = dino.student.state_dict()
            torch.save(state_dict, "{}/checkpoint_epoch{}.pth".format(
                CHECKPOINT_DIR, epoch))
            logging.info(f'Checkpoint {epoch} saved!')
    return


def visualize_dino():
    DIR = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/tmp"
    CHECKPOINT_FILE = f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/code/results/checkpoints/checkpoint_epoch99.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_IMG_DIR=f"{DIR}/pets/dogsImages"
    DATA_MASK_DIR=f"{DIR}/pets/dogsMasks"

    dataset = torchvision.datasets.MNIST(
        "/tmp", True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     (0.1307,), (0.3081,)
            # )
        ])
    )
    
    # dataset = DogsDataset(DATA_IMG_DIR, DATA_MASK_DIR)

    """
    dataset = torchvision.datasets.CIFAR10(
        "/tmp",
        True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Grayscale(),
            # torchvision.transforms.Normalize(
            #    (0.1307,), (0.3081,)
            # )
        ])
    )
    """

    loader_args = dict(batch_size=10, num_workers=os.cpu_count(), pin_memory=True)
    dataset = DataLoader(dataset, shuffle=True, **loader_args)

    # w_featmap = 28 // 7
    # h_featmap = 28 // 7
    k_size = 2
    w_featmap = 28 // k_size
    h_featmap = 28 // k_size
    # w_featmap = 104 // 8
    # h_featmap = 104 // 8
    dino = DINO()
    dino.student.to(DEVICE)
    dino.student.load_state_dict(torch.load(CHECKPOINT_FILE))
    dino.student.eval()
    for image_batch, labels in dataset:
    #for item in dataset:
        #print("Label: ", labels, image_batch.shape)
        #image_batch = item["image"]
        ib = image_batch
        image_batch = cut_image(image_batch)
        image_batch = image_batch.to(DEVICE)
        attn = dino.student.get_last_attention(image_batch)
        number_heads = attn.shape[1]
        
        """
        with torch.no_grad():
            val, idx = torch.sort(attn)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            threshold = 0.6
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(number_heads):
                th_attn[head] = th_attn[head][idx2[head]]
            print("SHP: ", th_attn[0].shape)
        """

        for i in range(image_batch.shape[0]):
            attentions = attn[i, :, 0, 1:].reshape(number_heads, -1)
            # print(attn[i, 0, 0, 1:].shape)
            # attentions = attn[i, 0, 0, 1:].reshape(number_heads, -1)
            attentions = attentions.reshape(number_heads, w_featmap, h_featmap)
            attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=k_size, mode="nearest")[0].cpu().detach().numpy()
            attentions_mean = np.mean(attentions, axis=0)
            img = (ib[i,:,:]*255).detach().float().cpu().numpy().astype(np.uint8).transpose(1,2,0).squeeze()
            plt.imsave(f"tmp/img{i}.png", img)
            row = [np.pad(attentions_mean, 2)]
            for j in range(number_heads):
                row.append(np.pad(attentions[j], 2))
            row = np.hstack(row)
            plt.imsave(f"tmp/{i}.jpg", arr=row, format='jpg')
        break
    return


def cut_image(image):
    # cuts = [torch.tensor_split(img, 4, 3)
    #         for img in torch.tensor_split(image, 4, 2)]
    # cuts = [torch.tensor_split(img, 7, 3)
    #         for img in torch.tensor_split(image, 7, 2)]
    # cuts = [torch.tensor_split(img, 13, 3)
    #         for img in torch.tensor_split(image, 13, 2)]
    cuts = [torch.tensor_split(img, 14, 3)
            for img in torch.tensor_split(image, 14, 2)]
    DD = []
    for LL in cuts:
        for i in LL:
            DD.append(i.unsqueeze(1))
    DD = torch.concatenate(DD, dim=1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
    # train_dino()
    visualize_dino()
    # visualize_dino1()

    # attn = Attention(500, 10)
    # attn.type(torch.float32)

    # block = Block(500, 10)
    # vit = ViT(7, 16, 500, 10, 10)

    # vit = ViT(500, 10, 10, 16, 7)
    # x = np.random.rand(1, 16, 7, 7)
    # x = torch.from_numpy(x).type(torch.float32)
    # vit(x)
    # print(cosine_scheduler(0.0005 * (8 / 256), 1E-6, 100, 1000, warmup_epochs=10, start_warmup_value=0))
