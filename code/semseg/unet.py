from pathlib import Path
import torch
from torchvision.transforms.functional import center_crop
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.nn.functional as F
from PIL import Image

class AttentionUNet(torch.nn.Module):
    def __init__(self) -> None:
        super(AttentionUNet, self).__init__()
        self.print_shapes = False
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 3)
        self.relu2 = torch.nn.ReLU()
        self.mp1 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(64, 128, 3)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(128, 128, 3)
        self.relu4 = torch.nn.ReLU()
        self.mp2 = torch.nn.MaxPool2d(2, 2)

        self.conv5 = torch.nn.Conv2d(128, 256, 3)
        self.relu5 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv2d(256, 256, 3)
        self.relu6 = torch.nn.ReLU()
        self.mp3 = torch.nn.MaxPool2d(2, 2)

        self.conv7 = torch.nn.Conv2d(256, 512, 3)
        self.relu7 = torch.nn.ReLU()
        self.conv8 = torch.nn.Conv2d(512, 512, 3)
        self.relu8 = torch.nn.ReLU()
        self.mp4 = torch.nn.MaxPool2d(2, 2)

        self.conv9 = torch.nn.Conv2d(512, 512, 3)
        self.relu9 = torch.nn.ReLU()
        self.conv10 = torch.nn.Conv2d(512, 512, 3)
        self.relu10 = torch.nn.ReLU()

        self.upConv1 = torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv11 = torch.nn.Conv2d(1024, 512, 3)
        self.relu11 = torch.nn.ReLU()
        self.conv12 = torch.nn.Conv2d(512, 256, 3)
        self.relu12 = torch.nn.ReLU()

        self.upConv2 = torch.nn.ConvTranspose2d(256, 256, 2, 2)
        self.conv13 = torch.nn.Conv2d(512, 256, 3)
        self.relu13 = torch.nn.ReLU()
        self.conv14 = torch.nn.Conv2d(256, 128, 3)
        self.relu14 = torch.nn.ReLU()

        self.upConv3 = torch.nn.ConvTranspose2d(128, 128, 2, 2)
        self.conv15 = torch.nn.Conv2d(256, 198, 3)
        self.relu15 = torch.nn.ReLU()
        self.conv16 = torch.nn.Conv2d(198, 64, 3)
        self.relu16 = torch.nn.ReLU()

        self.upConv4 = torch.nn.ConvTranspose2d(64, 64, 2, 2)
        self.conv17 = torch.nn.Conv2d(128, 64, 3)
        self.relu17 = torch.nn.ReLU()
        self.conv18 = torch.nn.Conv2d(64, 64, 3)
        self.relu18 = torch.nn.ReLU()

        self.conv19 = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):

        # Block1
        x = self.relu1(self.conv1(x))
        out1 = self.relu2(self.conv2(x))
        x = self.mp1(out1)

        if self.print_shapes:
            print("M1: ", x.shape)

        # Block2
        x = self.relu3(self.conv3(x))
        out2 = self.relu4(self.conv4(x))
        x = self.mp2(out2)

        if self.print_shapes:
            print("M2: ", x.shape)

        # Block3
        x = self.relu5(self.conv5(x))
        out3 = self.relu6(self.conv6(x))
        x = self.mp3(out3)

        if self.print_shapes:
            print("M3: ", x.shape)

        # Block4
        x = self.relu7(self.conv7(x))
        out4 = self.relu8(self.conv8(x))
        x = self.mp4(out4)

        if self.print_shapes:
            print("M4: ", x.shape)

        x = self.relu9(self.conv9(x))
        out5 = self.relu10(self.conv10(x))

        if self.print_shapes:
            print("Last: ", out5.shape)

        x = self.upConv1(out5)
        x = F.pad(x, (
            int((out4.shape[2]-x.shape[2])/2),
            int((out4.shape[2]-x.shape[2])/2),
            int((out4.shape[3]-x.shape[3])/2),
            int((out4.shape[3]-x.shape[3])/2)
            ), 'constant', 0)
        # out4Crop = self.crop(out4, 56, 56)
        # x = torch.concat([out4Crop, x], axis=1)
        x=torch.concat([out4, x], axis=1)
        x=self.relu11(self.conv11(x))
        x=self.relu12(self.conv12(x))

        if self.print_shapes:
            print("PUC1: ", x.shape)

        x=self.upConv2(x)
        x = F.pad(x, (
            int((out3.shape[2]-x.shape[2])/2) + 1,
            int((out3.shape[2]-x.shape[2])/2),
            int((out3.shape[3]-x.shape[3])/2) + 1,
            int((out3.shape[3]-x.shape[3])/2)
            ), 'constant', 0)
        # out3Crop = self.crop(out3, 104, 104)
        # x = torch.concat([out3Crop, x], axis=1)
        x=torch.concat([out3, x], axis=1)
        x=self.relu13(self.conv13(x))
        x=self.relu14(self.conv14(x))

        if self.print_shapes:
            print("PUC2: ", x.shape)

        x=self.upConv3(x)
        x = F.pad(x, (
            int((out2.shape[2]-x.shape[2])/2),
            int((out2.shape[2]-x.shape[2])/2),
            int((out2.shape[3]-x.shape[3])/2),
            int((out2.shape[3]-x.shape[3])/2)
            ), 'constant', 0)
        # out2Crop = self.crop(out2, 200, 200)
        # x = torch.concat([out2Crop, x], axis=1)
        x=torch.concat([out2, x], axis=1)
        x=self.relu15(self.conv15(x))
        x=self.relu16(self.conv16(x))

        if self.print_shapes:
            print("PU3: ", x.shape)

        x=self.upConv4(x)
        x = F.pad(x, (
            int((out1.shape[2]-x.shape[2])/2),
            int((out1.shape[2]-x.shape[2])/2),
            int((out1.shape[3]-x.shape[3])/2),
            int((out1.shape[3]-x.shape[3])/2)
            ), 'constant', 0)
        # out1Crop = self.crop(out1, 392, 392)
        # x = torch.concat([out1Crop, x], axis=1)
        x = torch.concat([out1, x], axis=1)
        x=self.relu17(self.conv17(x))
        x=self.relu18(self.conv18(x))

        x = F.pad(x, (
            4,4,4,4
            ), 'constant', 0)
        out = self.conv19(x)

        if self.print_shapes:
            print("out: ", out.shape)
        return out


class UNet(torch.nn.Module):
    def __init__(self) -> None:
        """
        Unet model for semantic segmentation. Input Size: 3 x 572 x 572
        Output: 388 x 388 x 2
        """
        self.print_shapes = False
        super(UNet, self).__init__()

        self.down_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.mp1 = torch.nn.MaxPool2d(2, 2)

        self.down_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.mp2 = torch.nn.MaxPool2d(2, 2)

        self.down_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.mp3 = torch.nn.MaxPool2d(2, 2)

        self.down_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.mp4 = torch.nn.MaxPool2d(2, 2)

        self.double_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv1 = torch.nn.ConvTranspose2d(512, 512, 2, 2)
        self.double_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv2 = torch.nn.ConvTranspose2d(256, 256, 2, 2)
        self.double_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv3 = torch.nn.ConvTranspose2d(128, 128, 2, 2)
        self.double_conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 198, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(198),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(198, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.upConv4 = torch.nn.ConvTranspose2d(64, 64, 2, 2)
        self.double_conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )

        self.conv19 = torch.nn.Conv2d(64, 1, 1)


    def crop(self, tensor, nHeight, nWidth):
        x, y = tensor.shape[-2:]
        center_x, center_y = x // 2, y // 2
        tX, tY = (center_x - nWidth // 2), center_y - nHeight // 2
        return tensor[:, :, tX:tX + nWidth, tY:tY + nHeight]

    def forward(self, x):

        out1 = self.down_block1(x)
        x = self.mp1(out1)

        if self.print_shapes:
            print("M1: ", x.shape)

        out2 = self.down_block2(x)
        x = self.mp2(out2)

        if self.print_shapes:
            print("M2: ", x.shape)

        out3 = self.down_block3(x)
        x = self.mp3(out3)

        if self.print_shapes:
            print("M3: ", x.shape)

        # Block4
        out4 = self.down_block4(x)
        x = self.mp4(out4)

        if self.print_shapes:
            print("M4: ", x.shape)

        out5 = self.double_conv1(x)

        if self.print_shapes:
            print("Last: ", out5.shape)

        x = self.upConv1(out5)
        diffY = out4.size()[2] - x.size()[2]
        diffX = out4.size()[3] - x.size()[3]
        x = F.pad(x, 
            [ diffX // 2, diffX - diffX // 2, 
              diffY // 2, diffY - diffY // 2 ]
        )
        x=torch.concat([out4, x], axis=1)
        x = self.double_conv2(x)

        if self.print_shapes:
            print("PUC1: ", x.shape)

        x=self.upConv2(x)
        diffY = out3.size()[2] - x.size()[2]
        diffX = out3.size()[3] - x.size()[3]
        x = F.pad(x, 
            [ diffX // 2, diffX - diffX // 2, 
              diffY // 2, diffY - diffY // 2 ]
        )
        x=torch.concat([out3, x], axis=1)
        x = self.double_conv3(x)

        if self.print_shapes:
            print("PUC2: ", x.shape)

        x=self.upConv3(x)
        diffY = out2.size()[2] - x.size()[2]
        diffX = out2.size()[3] - x.size()[3]
        x = F.pad(x, 
            [ diffX // 2, diffX - diffX // 2, 
              diffY // 2, diffY - diffY // 2 ]
        )
        x=torch.concat([out2, x], axis=1)
        x = self.double_conv4(x)

        if self.print_shapes:
            print("PU3: ", x.shape)

        x=self.upConv4(x)
        diffY = out1.size()[2] - x.size()[2]
        diffX = out1.size()[3] - x.size()[3]
        x = F.pad(x, 
            [ diffX // 2, diffX - diffX // 2, 
              diffY // 2, diffY - diffY // 2 ]
        )
        x = torch.concat([out1, x], axis=1)
        x = self.double_conv5(x)

        out = self.conv19(x)
        if self.print_shapes:
            print("out: ", out.shape)
        return out


def resize_images():
    DATA_DIR="/home/vishnu/Documents/EngProjs/Dataset"
    data=np.load(f"{DATA_DIR}/MNIST_large500.npy")
    output=f"{DATA_DIR}/MNIST500/Images"
    factor=388/500
    for i in range(data.shape[0]):
        img=cv2.resize(data[i], (0, 0), fx=factor, fy=factor)
        cv2.imwrite(f"{output}/Img{i}.jpg", img)


def gen_dogs():
    lists_file="/home/vishnu/Documents/EngProjs/Dataset/pets/Allannotations/list.txt"
    img_files_dir="/home/vishnu/Documents/EngProjs/Dataset/pets/Allimages"
    mask_files_dir="/home/vishnu/Documents/EngProjs/Dataset/pets/Allannotations/trimaps"
    out_img_files_dir="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsImages"
    out_mask_files_dir="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsMasks"
    with open(lists_file, 'r') as f:
        lines=[line for line in f.readlines()]
    lines=lines[6:]
    lines=[line for line in lines if ord(line[0]) >= ord('a')]
    for line in lines:
        f_name=line.split(" ")[0]
        shutil.copy(f'{img_files_dir}/{f_name}.jpg',
                    f'{out_img_files_dir}/{f_name}.jpg')
        shutil.copy(f'{mask_files_dir}/{f_name}.png',
                    f'{out_mask_files_dir}/{f_name}.jpg')


def update_masks():
    out_mask_files_dir="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsMasks"
    files=[
        f"{out_mask_files_dir}/{file}" for file in os.listdir(out_mask_files_dir)]
    for file in files:
        img=cv2.imread(file, cv2.COLOR_BGR2GRAY)
        img[img == 1] = 255
        img[img == 2] = 0
        img[img == 3] = 255
        cv2.imwrite(file, img)


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
        newW, newH = 150, 150
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
        mask = Image.open(mask_file_path)
        image = self.preprocess(image, False)
        mask = self.preprocess(mask, True)
        image=torch.from_numpy(image).type(torch.float32)
        mask=torch.from_numpy(mask).type(torch.float32)
        return {
            "image": image,
            "mask": mask
        }

def dice_loss(pred, mask):
    return 1 - dice_coefficient(pred, mask)

def dice_coefficient(pred, mask, reduce_batch_first=True):
    epsilon = 1E-6
    SM = (-1, -2) if reduce_batch_first else (-1, -2, -3)
    num = 2 * ((pred * mask).sum(SM))
    den = pred.sum(SM) + mask.sum(SM)
    den = torch.where(den == 0, num, den)
    dice = (num + epsilon) / (den + epsilon)
    return dice.mean()

def train_unet():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    DATA_IMG_DIR="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsImages"
    DATA_MASK_DIR="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsMasks"
    CHECKPOINT_DIR = f"results/checkpoints"
    VAL_PERCENT = 0.2
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 1E-3
    DEVICE = 'cpu'
    CHECKPOINT = "NEW"
    OPTIMIZER = "Adam"
    SAVE_CHECKPOINT = True

    dataset = DogsDataset(DATA_IMG_DIR, DATA_MASK_DIR)

    TRAINING_SIZE = int(len(dataset) * (1-VAL_PERCENT))
    VAL_SIZE = len(dataset) - TRAINING_SIZE

    train_set, val_set = random_split(
        dataset, [TRAINING_SIZE, VAL_SIZE],
        generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=BATCH_SIZE,
                       num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'''Starting training:
        Epochs:          {EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Training size:   {TRAINING_SIZE}
        Validation size: {VAL_SIZE}
        Checkpoints:     {CHECKPOINT}
        Device:          {DEVICE}
        Optimizer:       {OPTIMIZER}
    ''')

    model = UNet()
    model.to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        with tqdm(total=TRAINING_SIZE, desc=f'Epoch {epoch}/{EPOCHS}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(DEVICE)
                true_masks = true_masks.to(DEVICE)
                masks_pred = model(images)
                loss = loss_fn(masks_pred.squeeze(1), true_masks.squeeze(1).float())
                loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)),
                                  true_masks.squeeze(1).float())
                optimizer.zero_grad(set_to_none=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (TRAINING_SIZE // (5 * BATCH_SIZE))
                if division_step > 0:
                    if global_step % division_step == 0:
                        model.eval()
                        num_val_batches = len(val_loader)
                        dice_score = 0

                        for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                            image, mask_true = batch['image'], batch['mask']
                            image = image.to(DEVICE)
                            mask_true = mask_true.to(DEVICE)
                            # predict the mask
                            mask_pred = model(image)
                            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                            mask_true = torch.unsqueeze(mask_true, axis=1)
                            dice_score += dice_coefficient(mask_pred,
                                                           mask_true, reduce_batch_first=False)

                        dice_score /= max(num_val_batches, 1)
                        model.train()
                        scheduler.step(dice_score)
                        logging.info(
                            'Validation Dice score: {}'.format(dice_score))

        if SAVE_CHECKPOINT:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, "{}/checkpoint_epoch{}.pth".format(
                CHECKPOINT_DIR, epoch))
            logging.info(f'Checkpoint {epoch} saved!')
    return


def pred(model_file, images_data_dir, masks_data_dir):
    images_data_dir=images_data_dir
    masks_data_dir=masks_data_dir
    image_files=os.listdir(images_data_dir)
    image_file=image_files[45]
    image_file_path=f"{images_data_dir}/{image_file}"
    mask_file_path=f"{masks_data_dir}/{image_file}"
    image = Image.open(image_file_path)
    mask = Image.open(mask_file_path)
    image = DogsDataset.preprocess(image, False)
    mask = DogsDataset.preprocess(mask, True)
    img = torch.from_numpy(image).type(torch.float32)
    # mask = torch.from_numpy(mask).type(torch.float32)
    # img = torch.unsqueeze(img, 0)

    # img = (255.0*img[0]).type(torch.uint8)
    # img = img.numpy()
    # img = np.moveaxis(img, 0, -1)

    # mask = (255.0*mask).type(torch.uint8)
    # mask = mask.numpy()
    # mask = np.moveaxis(mask, 0, -1)
    
    # cv2.imwrite("test/img.jpg", img)
    # cv2.imwrite("test/pred.jpg", mask)
    # return
    
    img = torch.from_numpy(image).type(torch.float32)
    img = torch.unsqueeze(img, 0)
    # mask=torch.from_numpy(mask).type(torch.float32)

    model = UNet()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    out = model(img)

    img = (255.0*img[0]).type(torch.uint8)
    img = img.numpy()
    img = np.moveaxis(img, 0, -1)

    out = torch.sigmoid(out) > 0.5
    out = (255.0*out[0]).type(torch.uint8)
    out = out.numpy()
    out = np.moveaxis(out, 0, -1)
    cv2.imwrite("test/img.jpg", img)
    cv2.imwrite("test/pred.jpg", out)

if __name__ == "__main__":
    # gen_dogs()
    # update_masks()
    # train_unet()
    
    DATA_IMG_DIR="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsImages"
    DATA_MASK_DIR="/home/vishnu/Documents/EngProjs/Dataset/pets/dogsMasks"
    pred("results/checkpoints/checkpoint_epoch2.pth", DATA_IMG_DIR, DATA_MASK_DIR)