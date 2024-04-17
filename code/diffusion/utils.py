import os
import cv2
import logging
from model import UNet, SinPositionEmbedding, SelfAttentionBlock
import torchvision
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np


def loss_fn(e, e_pred):
    return ((e - e_pred)**2).mean()


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


def sample():
    SAMPLE_SIZE = [3, 32, 32]
    DIMENSIONS = SAMPLE_SIZE[0] * SAMPLE_SIZE[1] * SAMPLE_SIZE[1]
    BETA_INITIAL = 10E-4
    BETA_FINAL = 0.02
    T = 1000

    model = UNet()
    # model.type(torch.bfloat16)

    X_t = torch.from_numpy(np.random.multivariate_normal(
            [0 for i in range(DIMENSIONS)], 
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE))#.type(torch.bfloat16)
    beta_values = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    for t in range(T, -1, -1):
        beta_t = next(beta_values)
        alpha_t = 1 - beta_t
        sigma_t = beta_t

        # z = torch.from_numpy(np.random.multivariate_normal(
        #     [0 for i in range(DIMENSIONS)], 
        #     np.identity(DIMENSIONS)
        # ).reshape(SAMPLE_SIZE)).type(torch.bfloat16) if t > 1 else torch.from_numpy(np.zeros(SAMPLE_SIZE)).type(torch.bfloat16)
        z = torch.from_numpy(np.random.multivariate_normal(
            [0 for i in range(DIMENSIONS)], 
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE)) if t > 1 else torch.from_numpy(np.zeros(SAMPLE_SIZE))
        z = z.unsqueeze(0)

        factor = (1 - alpha_t) / ((1-alpha_t) **0.5)
        pred = model(X_t.unsqueeze(0))
        Xo = factor * pred
        Xo =  (1 / (alpha_t)**0.5) * Xo
        Xo += (sigma_t*z)
    return Xo

def get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T):
    vals = {}
    beta_values = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    vals[0] = 1 - next(beta_values)
    for i,val in enumerate(beta_values):
        vals[i+1] = vals[i] * (1 - val)
    return vals

def train(train_params, model):
    DIR = "/Users/vishnudevarakonda/Documents/EngProjs/websites/DevarakondaV.github.io/tmp" # os.environ['WORKSPACE_DIR']
    TMP_DIR = f'{DIR}/tmp/'
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    CHECKPOINT_DIR = f"{TMP_DIR}/results/diffusion/checkpoints"
    BATCH_SIZE = 16
    LEARNING_RATE = 2E-4
    DEVICE = 'cpu'
    CHECKPOINT = "NEW"
    OPTIMIZER = "Adam"
    SAVE_CHECKPOINT = True
    SAMPLE_SIZE = train_params["sample_size"]
    BETA_INITIAL = train_params["beta_initial"]
    BETA_FINAL = train_params["beta_final"]
    T = train_params["T"]
    ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    DIMENSIONS = SAMPLE_SIZE[0] * SAMPLE_SIZE[1] * SAMPLE_SIZE[1]

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
            torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stop = len(dataset) * 1000
    minLoss = float('inf')
    checkpoint_number = 0
    i = 0
    while True:
        # Xo = np.identity(DIMENSIONS)
        Xo,l = dataset[np.random.randint(0, len(dataset))]
        # print(Xo.shape)
        # t_step = np.random.uniform(0, T, size = 1)
        t_step = np.random.randint(0, T)
        alpha_bar_t = ALPHA_BAR_VALUES[t_step]
        I = np.identity(DIMENSIONS)
        e = np.random.multivariate_normal(
            np.zeros(DIMENSIONS),
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE)
        model_input = ((alpha_bar_t ** 0.5) * Xo) + ((1 - alpha_bar_t)**0.5) * e
        model_input = model_input.type(torch.float32).unsqueeze(0)
        e_pred = model(model_input, t_step).squeeze(0)
        e = torch.from_numpy(e)
        loss = loss_fn(e , e_pred)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < minLoss:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict=model.state_dict()
            torch.save(state_dict, "{}/checkpoint_epoch0.pth".format(
                CHECKPOINT_DIR, checkpoint_number))
            logging.info(f'Checkpoint {checkpoint_number} saved!')
            minLoss = loss.item()
            checkpoint_number += 1
        if stop == i:
            break
        i += 1
    return


def sample(train_params, model):
    SAMPLE_SIZE = train_params["sample_size"]
    DIMENSIONS = SAMPLE_SIZE[0] * SAMPLE_SIZE[1] * SAMPLE_SIZE[1]
    BETA_INITIAL = 10E-4
    BETA_FINAL = 0.02
    T = train_params["T"]
    BETA_SCHEDULER = [i for i in linear_scheduler(BETA_INITIAL, BETA_FINAL, T)]
    Xprev = np.random.multivariate_normal(
        np.zeros(DIMENSIONS),
        np.identity(DIMENSIONS)
    ).reshape(SAMPLE_SIZE)
    model.eval()
    for t in range(T):
        print("T: ", t)
        if t > 0:
            z = np.random.multivariate_normal(np.zeros(DIMENSIONS), np.identity(DIMENSIONS)).reshape(SAMPLE_SIZE)
        else:
            z = 0
        beta_t = BETA_SCHEDULER[t]
        alpha_t = 1 - beta_t
        sigma_t = 0
        e_pred = model(
            # torch.from_numpy(Xprev).type(torch.float16).unsqueeze(0),
            torch.from_numpy(Xprev).type(torch.float32).unsqueeze(0),
            t
        ).squeeze(axis=0).detach().float().numpy()
        scale_factor = (1 - alpha_t)/((1-alpha_t)**0.5)
        Xn = ((1/(alpha_t**0.5)) * (Xprev - scale_factor * e_pred)) + sigma_t * z
        Xprev = Xn # .squeeze(axis=0).detach().float().numpy()
        if t == (T-1) or t % 50 == 0:
            s_img = (Xn*255).astype(np.uint8).squeeze()
            cv2.imwrite(f"/Users/vishnudevarakonda/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/sample{t}.jpg", s_img)
    return Xn


def get_diffusion_model_train_params(
        T: int,
        sample_size: list,
        beta_initial: float,
        beta_final: float
    ):
    return {
        "sample_size": sample_size,
        "beta_initial": beta_initial,
        "beta_final": beta_final,
        "T": T
    }

if __name__ == "__main__":
    # pE = SinPositionEmbedding(10, 10)
    # pE.eval()
    # print(pE(1))
    # block = SelfAttentionBlock(10, 5)
    # print(block(torch.zeros(1, 2, 5), 0))
    # exit()
    train_params = get_diffusion_model_train_params(
        T = 1000,
        # sample_size = [1, 28, 28],
        sample_size = [1, 32, 32],
        beta_initial = 10E-4,
        beta_final = 0.02
    )
    # beta = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    # for item in beta:
    #     print(item)
    # ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    # print(ALPHA_BAR_VALUES)
    model = UNet(
        attn_dimension=512 * 2 * 2, #512
    )
    # model.type(torch.bfloat16)

    model.state_dict(torch.load("/Users/vishnudevarakonda/Documents/EngProjs/websites/DevarakondaV.github.io/tmp/tmp/results/diffusion/checkpoints/checkpoint_epoch0.pth"))
    x = sample(train_params, model)

    # train(train_params, model)

