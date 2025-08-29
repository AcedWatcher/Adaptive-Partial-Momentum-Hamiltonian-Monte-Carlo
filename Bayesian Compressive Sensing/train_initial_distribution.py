import os

import numpy as np
import torch
import torch.nn as nn
import sys
import normflows as nf
from tqdm import tqdm
import hydra
import subprocess

from core.target import HorseshoeTarget

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.chdir(path)
    else:
        os.chdir(path)

def main():

    path = ''
    data_path = ''
    mkdir(path + '')

    alpha = 0

    d = 64
    sigma_0 = 0.005
    tau = 0.01

    seed = 0
    anneal_steps = 30
    inner_iters = 1000
    sigma_0 = 0.005
    batch_size = 128
    save_interval = 10

    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, true_w, true_lamb = np.load(data_path + 'X_train.npy'), \
        np.load(data_path + 'y_train.npy'), \
        np.load(data_path + 'w.npy'), \
        np.load(data_path + 'lambda.npy')

    X_train, y_train, true_w, true_lamb = torch.from_numpy(X_train).double(), \
        torch.from_numpy(y_train).double(), \
        torch.from_numpy(true_w).double(), \
        torch.from_numpy(true_lamb).double()

    target = HorseshoeTarget(X_train, y_train, tau, sigma_0)

    initial_dist = nf.core.NormalizingFlow(
        q0=nf.distributions.DiagGaussian(
            shape=2 * d
        ),
        flows=None,
        p=target
    ).double()

    # Train initial distribution
    inner_iters = inner_iters
    anneal_steps = anneal_steps
    initial_dist_losses = np.zeros((anneal_steps * inner_iters))
    initial_dist_log_scales = np.zeros((anneal_steps * inner_iters, 2 * d))
    initial_dist_means = np.zeros((anneal_steps * inner_iters, 2 * d))
    learning_rates = [1 * 10 ** (-x) for x in np.linspace(1, 3, anneal_steps)]
    for i in tqdm(range(1, anneal_steps + 1)):
        target.gamma = 0.8 ** (anneal_steps - i)
        initial_dist_optimizer = torch.optim.Adam(initial_dist.parameters(),
                                                  lr=learning_rates[i - 1])

        for k in range(inner_iters):
            initial_dist_optimizer.zero_grad()

            if alpha == 0:
                loss = initial_dist.reverse_kld(num_samples=batch_size)
            else:
                loss = initial_dist.reverse_alpha_div(
                    num_samples=batch_size,
                    alpha=alpha)

            loss.backward()
            initial_dist_optimizer.step()

            initial_dist_losses[(i - 1) * inner_iters + k] = loss.detach().numpy()
            initial_dist_log_scales[(i - 1) * inner_iters + k, :] = initial_dist.q0.log_scale.detach().numpy()
            initial_dist_means[(i - 1) * inner_iters + k, :] = initial_dist.q0.loc.detach().numpy()

        if i % save_interval == 0:
            np.save('losses.npy', initial_dist_losses)
            np.save('log_scales.npy', initial_dist_log_scales)
            np.save('means.npy', initial_dist_means)
            torch.save(initial_dist.state_dict(), 'initial_dist_{}.pt'.format(i))

    # Final saving
    np.save('losses.npy', initial_dist_losses)
    np.save('log_scales.npy', initial_dist_log_scales)
    np.save('means.npy', initial_dist_means)
    torch.save(initial_dist.state_dict(), 'initial_dist_final.pt')

    print("loss:",initial_dist_losses)
    print("means:",initial_dist_means)
    #print('log_scales:',initial_dist_log_scales)


def log_qz(z, mean, var):
    k, dim = z.shape
    mean = mean.repeat(k, 1)
    var = var.unsqueeze(1)
    var_inv = (1 / var) * torch.eye(dim)
    det = torch.prod(var)
    x = z - mean
    res = torch.matmul(x, var_inv)
    res = torch.matmul(res, x.T)
    log_q = torch.log(1 / torch.sqrt(torch.tensor(np.pi)) ** dim * det) + torch.diag(res)
    return log_q


if __name__ == "__main__":
    main()
