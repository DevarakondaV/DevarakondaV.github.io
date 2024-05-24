import os
import cv2
import logging
from model import UNet, SinPositionEmbedding, SelfAttentionBlock, get_timestep_embedding
import torchvision
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = torch.device("cuda")
DTYPE = torch.float32

# def loss_fn(e, e_pred):
#    return ((e - e_pred)**2).mean() # (dim=(1,2,3))


def linear_scheduler(initial: float, final: float, T: int):
    """
    schedules value linearly from initial to final.

    args:
        initial: The initial value.
        final: The final value.
        T: The number of values from initial to final.
    """
    for i in range(0, T):
        yield initial + i * (final - initial) / T

def get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T):
    vals = {}
    beta_values = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    vals[0] = 1 - next(beta_values)
    for i,val in enumerate(beta_values):
        vals[i+1] = vals[i] * (1 - val)
    return vals

def train(train_params, model):
    DIR = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp" # os.environ['WORKSPACE_DIR']
    TMP_DIR = f'{DIR}'
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    CHECKPOINT_DIR = f"{TMP_DIR}/results/diffusion/checkpoints"
    BATCH_SIZE = train_params["batch_size"]
    LEARNING_RATE = train_params["learning_rate"] # 1E-4
    CHECKPOINT = "NEW"
    OPTIMIZER = "Adam"
    BETA_INITIAL = train_params["beta_initial"]
    BETA_FINAL = train_params["beta_final"]
    EPOCH = train_params["epoch"]
    T = train_params["T"]
    ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)

    # dataset = torchvision.datasets.CIFAR10(
    #     "/tmp",
    #     train=True,
    #     transforms=torchvision.transforms.Compose([
    #         torchvision.transforms.toTensor(),
    #         torchvision.transforms.Normalize([
    #             (0, 0), (0, 0)
    #         ])
    #     ])
    # )

    dataset = torchvision.datasets.MNIST(
        "/tmp",
        True,
        download=True,
        transform=torchvision.transforms.Compose([
            # torchvision.transforms.Pad(2),
            # torchvision.transforms.Resize((32, 32), torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0, 1),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
    
    loader_args = dict(batch_size=BATCH_SIZE,
                       num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'''Starting training:
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Checkpoints:     {CHECKPOINT}
        Device:          {DEVICE}
        Optimizer:       {OPTIMIZER}
    ''')

    model.to(DEVICE)
    # print([name for name,_ in model.named_parameters()])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.9
    )

    """
    img,_ = dataset[0]
    # img = img * 255
    for t in range(0, T):
        e = torch.randn_like(img).numpy()
        alpha_bar_t = ALPHA_BAR_VALUES[t]
        print(alpha_bar_t, e.mean())
        model_input = ((alpha_bar_t ** 0.5) * img.numpy()) + ((1 - alpha_bar_t)**0.5) * e
        if t == 0 or t == (T-1):
            plt.figure(t)
            plt.hist((model_input).flatten())
            plt.savefig(f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/{t}_hist.jpg")
            # fig, ax0 = plt.subplots(1, 1)
            # ax0.hist(s_img.flatten())
            # plt.savefig()
        # if t % (T//50) == 0:
        # s_img = (model_input*255).cpu().detach().float().numpy().astype(np.uint8).squeeze()
        # s_img = (model_input).astype(np.uint8).squeeze()
        print(model_input.mean(), model_input.std())
        s_img = (model_input*255).squeeze()
        cv2.imwrite(f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/sample{t}.jpg", s_img)
        # break
    return
    """
    train_loader = list(train_loader)[:25]

    for epoch in range(EPOCH):
        epoch_loss = 0
        model.train()
        for i, (data, y) in enumerate(train_loader):
            c = data.shape[0]
            t_step = torch.randint(0, T, size=(c // 2 + 1,)) # np.random.randint(0, T, size = 1)[0]
            t_step = torch.cat([t_step, T-t_step-1], dim=0)[:c]

            alpha_bar_t = torch.from_numpy(np.array([ALPHA_BAR_VALUES[p.item()] for p in t_step])).reshape(c, 1, 1, 1)
            e = torch.randn_like(data)
            model_input = (alpha_bar_t.sqrt() * data) + (((1 - alpha_bar_t) ** 0.5) * e)
            t_step = torch.from_numpy(np.array(t_step)).to(DEVICE).type(DTYPE)
            
            
            model_input = model_input.to(DEVICE).type(DTYPE)
            e = e.to(DEVICE).type(DTYPE)
            e_pred = model(model_input, t_step)
            loss = (e-e_pred).square().sum(dim=(1,2,3)).mean(dim=0) # torch.nn.functional.mse_loss(e, e_pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logging.info(f'Epoch: {epoch}/{EPOCH}, i: {i}/{len(train_loader)}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]["lr"]}')
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        logging.info(f'EPOCH LOSS: {epoch_loss}')
        scheduler.step(epoch_loss)

        Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        state_dict=model.state_dict()
        torch.save(state_dict, "{}/checkpoint_epoch{}.pth".format(
            CHECKPOINT_DIR, epoch))
        logging.info(f'Checkpoint {epoch} saved!, Loss: {loss.item()}')
        model.eval()
        sample(train_params, model)


def sample(train_params, model):
    SAMPLE_SIZE = train_params["sample_size"]
    DIMENSIONS = SAMPLE_SIZE[0] * SAMPLE_SIZE[1] * SAMPLE_SIZE[1]
    BETA_INITIAL = train_params["beta_initial"]
    BETA_FINAL = train_params["beta_final"]
    T = train_params["T"]
    BETA_SCHEDULER = [i for i in linear_scheduler(BETA_INITIAL, BETA_FINAL, T)]
    ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    Xprev = torch.randn([1, 1, 28, 28])
    model.eval()
    for t in range(T-1, -1, -1):
        # print("T: ", t)
        if t > 0:
            z = torch.randn_like(Xprev)
        else:
            z = 0
        beta_t = BETA_SCHEDULER[t]
        alpha_t = 1 - beta_t
        alpha_bar_t = ALPHA_BAR_VALUES[t]
        sigma_t = beta_t**0.5
        Xprev = Xprev.to(DEVICE).type(DTYPE)
        e_pred = model(
            Xprev,
            torch.from_numpy(np.array([t])).to(DEVICE).type(DTYPE)
        )
        e_pred = e_pred.cpu().detach().float()
        scale_factor = (1 - alpha_t)/((1-alpha_bar_t)**0.5)
        Xprev = Xprev.cpu()
        Xn = ((1/(alpha_t**0.5)) * (Xprev - scale_factor * e_pred)) + sigma_t * z
        Xprev = Xn
        # s_img = (Xn*255).numpy().astype(np.uint8).squeeze()
        s_img = (Xn*255).numpy().squeeze()
        cv2.imwrite(f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/sample{t}.jpg", s_img)
        if t == 0 or t == (T-1):
            plt.figure(t)
            plt.hist((Xn).flatten())
            plt.savefig(f"/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/{t}_hist.jpg")
    return Xn


def get_diffusion_model_train_params(
        T: int,
        sample_size: list,
        beta_initial: float,
        beta_final: float,
        batch_size: float,
    ):
    return {
        "sample_size": sample_size,
        "beta_initial": beta_initial,
        "beta_final": beta_final,
        "T": T,
        "learning_rate": 1E-4,#E-4,
        "epoch": 100,
        "batch_size": batch_size
    }

def make_joined_images():
    tmp_dirs = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp"
    image_dirs = [f'{tmp_dirs}/{i}' for i in range(10)]
    
    """
    zeros = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/0/{i}')) for i in os.listdir(f'{tmp_dirs}/0')]
    ones = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/1/{i}')) for i in os.listdir(f'{tmp_dirs}/1')]
    twos = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/2/{i}')) for i in os.listdir(f'{tmp_dirs}/2')]
    threes = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/3/{i}')) for i in os.listdir(f'{tmp_dirs}/3')]
    fours = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/4/{i}')) for i in os.listdir(f'{tmp_dirs}/4')]
    fives = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/5/{i}')) for i in os.listdir(f'{tmp_dirs}/5')]
    sixs = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/6/{i}')) for i in os.listdir(f'{tmp_dirs}/6')]
    sevens = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/7/{i}')) for i in os.listdir(f'{tmp_dirs}/7')]
    eights = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/8/{i}')) for i in os.listdir(f'{tmp_dirs}/8')]
    nines = [(int(i.split(".")[0].split("sample")[1]), cv2.imread(f'{tmp_dirs}/9/{i}')) for i in os.listdir(f'{tmp_dirs}/9')]
    """

    zeros = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/0/{i}')) for i in os.listdir(f'{tmp_dirs}/0')]
    ones = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/1/{i}')) for i in os.listdir(f'{tmp_dirs}/1')]
    twos = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/2/{i}')) for i in os.listdir(f'{tmp_dirs}/2')]
    threes = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/3/{i}')) for i in os.listdir(f'{tmp_dirs}/3')]
    fours = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/4/{i}')) for i in os.listdir(f'{tmp_dirs}/4')]
    fives = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/5/{i}')) for i in os.listdir(f'{tmp_dirs}/5')]
    sixs = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/6/{i}')) for i in os.listdir(f'{tmp_dirs}/6')]
    sevens = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/7/{i}')) for i in os.listdir(f'{tmp_dirs}/7')]
    eights = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/8/{i}')) for i in os.listdir(f'{tmp_dirs}/8')]
    nines = [(int(i.split(".")[0][6:]), cv2.imread(f'{tmp_dirs}/9/{i}')) for i in os.listdir(f'{tmp_dirs}/9')]

    # zeros = sorted(zeros, reverse=True, key = lambda x : x[0])
    images = {i:[] for i in range(100)}
    for i, img in zeros:
        images[i].append(img)
    for i, img in ones:
        images[i].append(img)
    for i, img in twos:
        images[i].append(img)
    for i, img in threes:
        images[i].append(img)
    for i, img in fours:
        images[i].append(img)
    for i, img in fives:
        images[i].append(img)
    for i, img in sixs:
        images[i].append(img)
    for i, img in sevens:
        images[i].append(img)
    for i, img in eights:
        images[i].append(img)
    for i, img in nines:
        images[i].append(img)

    images = {key:np.vstack([np.hstack(value[:5]), np.hstack(value[5:])]) for key, value in images.items()}
    for key, value in images.items():
        cv2.imwrite(f'{tmp_dirs}/images/{key}.jpg', value)

def make_video():
    tmp_dirs = "/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp"
    images_dir = f'{tmp_dirs}/images'
    images = [(int(i.split(".")[0]), cv2.imread(f'{images_dir}/{i}')) for i in os.listdir(images_dir) if i != "gif"]
    images = sorted(images, reverse=True, key = lambda x: x[0])
    images = [Image.fromarray(img) for _,img in images]
    for i in range(100):
        images.append(images[-1])
    img = images[0]
    img.save(f'{images_dir}/gif/diffusion.gif', format='GIF', append_images=images[1:], save_all=True, duration=10,loop=0)


if __name__ == "__main__":
    # make_joined_images()
    make_video()
    exit()
    """
    pE = SinPositionEmbedding(10, 10)
    pE.eval()
    print(pE(1))
    # block = SelfAttentionBlock(10, 5)
    # print(block(torch.zeros(1, 2, 5), 0))
    
    
    print(get_timestep_embedding(torch.from_numpy(np.array([3])), 10))
    exit()
    """

    train_params = get_diffusion_model_train_params(
        T = 100,
        sample_size = [1, 28, 28],
        beta_initial=10E-4,
        beta_final=0.02,
        batch_size=64
    )
    # beta = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    # for item in beta:
    #     print(item)
    # ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    # print(ALPHA_BAR_VALUES)
    model = UNet(
        attn_dimension=512,
        DEVICE=DEVICE, DTYPE=DTYPE
    )
    """
    model.to(DEVICE).type(DTYPE)
    out = model(
        torch.from_numpy(np.zeros((1, 1, 28, 28))).to(DEVICE).type(DTYPE),
        torch.from_numpy(np.array([0])).to(DEVICE).type(DTYPE)
    )
    print(out.shape)
    exit()
    """
    model.type(DTYPE).to(DEVICE)
    # model.type(torch.bfloat16)

    # train(train_params, model)

    """
    for param in torch.load("/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/results/diffusion/checkpoints/checkpoint_epoch937.pth"):
        print(param)
    print("PRINTING MODEL")
    for param in model.named_parameters():
        print(param[0])
    exit()
    """

    # 82
    # 90
    model.load_state_dict(torch.load("/home/vishnu/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/results/diffusion/checkpoints/checkpoint_epoch75.pth"))
    x = sample(train_params, model)
