# %%
import os
import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import normflows as nf
from tqdm import tqdm
import hydra
import subprocess

from core.target import HorseshoeTarget
from core.utils import SKSD
from core.models import HMC


def log_q(state, mean, log_var):
    var = torch.exp(log_var)
    var_inv = (1/var).unsqueeze(2) * torch.eye(var.shape[1]).to(state.device)
    det = var.prod(1)
    x = (state - mean).reshape(-1, mean.shape[1])
    res = torch.einsum('ij, ijl->il', x, var_inv)
    res = torch.einsum('ij, ij->i', res, x)
    log_qz = res - 0.5 * var.shape[1].value * torch.log(2 * torch.tensor(np.pi)) - 0.5 * log_var.sum(1)
    return log_qz

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.chdir(path)
    else:
        os.chdir(path)

def main():
    path = ''
    data_path = path + ''
    initial_name = ''

    initial_dist_path = ''

    model_name = initial_name + ''

    mkdir(path + '\\' + model_name)
    train_scale = False

    log_stepsize_min = -6
    log_stepsize_max = -5
    log_stepsize_const = None
    log_mass_min = 2
    log_mass_max = 3
    log_mass_const = None
    num_samples_estimate_initial_mean = 10000

    num_iters = 1000
    batch_size = 128
    hmc_lr = 0.01
    hmc_scale_lr = 0.005
    hmc_g_lr = 0.01
    save_interval = 100

    iters = num_iters

    d = 64
    sigma_0 = 0.005
    tau = 0.01
    num_hmc_steps = 20
    num_leapfrog_steps = 5
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)



    print("Loading observed data from", data_path)
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
    print("Loading initial dist from", initial_dist_path)
    initial_dist.load_state_dict(torch.load(initial_dist_path))

    initial_dist_samples, _ = initial_dist.sample(num_samples_estimate_initial_mean)
    initial_dist_mean = torch.mean(initial_dist_samples, dim=0).detach()

    if log_stepsize_const is None:
        print("Initializing with a range of stepsizes")
        hmc = HMC(target, 2 * d, initial_dist,
                  num_hmc_steps, num_leapfrog_steps,
                  log_stepsize_min, log_stepsize_max,
                  log_mass_min, log_mass_max,
                  train_scale, initial_dist_mean).double()
    else:
        print("Initializing with const stepsize and mass")
        hmc = HMC(target, 2 * d, initial_dist,
                  num_hmc_steps, num_leapfrog_steps,
                  log_stepsize_const, log_stepsize_const,
                  log_mass_const, log_mass_const,
                  train_scale, initial_dist_mean).double()

    hmc_optim = torch.optim.Adam(hmc.get_hmc_params(), lr=hmc_lr)
    if train_scale:
        print("Training with SKSD")
        scale_optim = torch.optim.Adam([hmc.scale], lr=hmc_scale_lr)
        g_optim = torch.optim.Adam([hmc.raw_g], lr=hmc_g_lr)
    else:
        print("Training without SKSD")

    losses = np.zeros((iters,))
    log_step_sizes = np.zeros((iters, num_hmc_steps, 2 * d))
    log_masses = np.zeros((iters, num_hmc_steps, 2 * d))
    acc_probs = np.zeros((iters,))

    torch.save(hmc.state_dict(), 'pre_training_hmc_model.pt')

    print("\n Training HMC params")


    start_time = time.time()

    for i in tqdm(range(iters)):
        hmc_optim.zero_grad()

        samples, ap = hmc.forward(batch_size)

        loss = - torch.mean(target.log_prob(samples))
        loss = 100 * loss


        loss.backward(retain_graph=train_scale)
        hmc_optim.step()

        if train_scale:
            scale_optim.zero_grad()
            g_optim.zero_grad()
            gradlogp = hmc.flows[0].gradlogP(samples)
            sksd = SKSD(samples, gradlogp, hmc.get_g())
            sksd.backward()
            scale_optim.step()
            hmc.raw_g.grad = -hmc.raw_g.grad  # Since we want to max sksd wrt g
            g_optim.step()

        losses[i] = loss.detach().numpy()
        log_step_sizes[i, :, :], log_masses[i, :, :] = hmc.get_np_params()
        acc_probs[i] = np.median(np.array(ap))

        if (i + 1) % save_interval == 0:
            torch.save(hmc.state_dict(), 'hmc_model_{}.pt'.format(i + 1))
            np.save('losses.npy', losses)
            np.save('log_step_sizes.npy', log_step_sizes)
            np.save('log_masses.npy', log_masses)
            np.save('acc_probs.npy', acc_probs)



    torch.save(hmc.state_dict(), 'final_hmc_model.pt')
    np.save('losses.npy', losses)
    np.save('log_step_sizes.npy', log_step_sizes)
    np.save('log_masses.npy', log_masses)
    np.save('acc_probs.npy', acc_probs)

    fin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_time = time.time()  - start_time
    print(hmc.state_dict())
    print(model_name + ' time: {}'.format(total_time))
    print(fin_time)

if __name__ == "__main__":
    main()
