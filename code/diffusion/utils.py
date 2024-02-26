import os
import logging
from model import UNet
import torchvision
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np


def loss_fn(e, e_pred):
    return (e - e_pred)**2


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
    model.type(torch.bfloat16)

    X_t = torch.from_numpy(np.random.multivariate_normal(
            [0 for i in range(DIMENSIONS)], 
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE)).type(torch.bfloat16)
    beta_values = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    for t in range(T, -1, -1):
        beta_t = next(beta_values)
        alpha_t = 1 - beta_t
        sigma_t = beta_t

        z = torch.from_numpy(np.random.multivariate_normal(
            [0 for i in range(DIMENSIONS)], 
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE)).type(torch.bfloat16) if t > 1 else torch.from_numpy(np.zeros(SAMPLE_SIZE)).type(torch.bfloat16)
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

def train():
    DIR = os.environ['WORKSPACE_DIR']
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
    SAMPLE_SIZE = [3, 32, 32]
    DIMENSIONS = SAMPLE_SIZE[0] * SAMPLE_SIZE[1] * SAMPLE_SIZE[1]
    BETA_INITIAL = 10E-4
    BETA_FINAL = 0.02
    ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    T = 1000

    dataset = torchvision.datasets.CIFAR10(
        "/tmp",
        train=True,
        transforms=torchvision.transforms.Compose([
            torchvision.transforms.toTensor(),
            torchvision.transforms.Normalize([
                (0, 0), (0, 0)
            ])
        ])
    )

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

    model = UNet()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stop = True
    minLoss = float('inf')
    checkpoint_number = 0
    while True:
        Xo = np.identity(DIMENSIONS)
        t_step = np.random.uniform(0, T, size = 1)

        alpha_bar_t = ALPHA_BAR_VALUES[t_step]

        I = np.identity(DIMENSIONS)
        
        e = np.random.multivariate_normal(
            DIMENSIONS, 
            np.identity(DIMENSIONS)
        ).reshape(SAMPLE_SIZE)

        model_input = ((alpha_bar_t ** 0.5) * Xo) + ((1 - alpha_bar_t)**0.5) * e

        e_pred = model(model_input, t_step)
        loss = loss_fn(e , e_pred)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < minLoss:
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict=model.state_dict()
            torch.save(state_dict, "{}/checkpoint_epoch{}.pth".format(
                CHECKPOINT_DIR, checkpoint_number))
            logging.info(f'Checkpoint {checkpoint_number} saved!')
            minLoss = loss.item()
            checkpoint_number += 1
        if stop:
            break
    return



if __name__ == "__main__":
    BETA_INITIAL = 10E-4
    BETA_FINAL = 0.02
    T = 1000
    # beta = linear_scheduler(BETA_INITIAL, BETA_FINAL, T)
    # for item in beta:
    #     print(item)
    # ALPHA_BAR_VALUES = get_alpha_bar_values(BETA_INITIAL, BETA_FINAL, T)
    # print(ALPHA_BAR_VALUES)
    sample()