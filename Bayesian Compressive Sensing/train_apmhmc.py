# %%
import datetime
import os
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
from core.models import APMHMC




def log_qz(z,mean,var):
    k, dim = z.shape
    mean = mean.repeat(k,1)
    var = var.unsqueeze(1)
    var_inv = (1/var) * torch.eye(dim)
    det = torch.prod(var)

    x = z - mean
    res = torch.matmul(x, var_inv)
    res = torch.matmul(res, x.T)
    log_q = torch.log(1/torch.sqrt(torch.tensor(np.pi))**dim * det) + torch.diag(res)
    return log_q

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
    initial_name = ''
    initial_dist_path = ''

    r1 = 0.1
    s = 0.9
    a = 0.9
    b = 20
    model_name = initial_name + '-scd'

    mkdir(path + '\\' + model_name)
    train_scale = False

    num_hmc_steps = 20
    num_leapfrog_steps = 5

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

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)


    print("Training APMHMC with SCD")

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
    initial_dist_var = torch.var(initial_dist_samples, dim=0).detach()

    def log_q(z):
        return log_qz(z, initial_dist_mean, initial_dist_var)

    if log_stepsize_const is None:
        print("Initializing with a range of stepsizes")
        hmc = APMHMC(target, 2 * d, initial_dist,
                  num_hmc_steps, num_leapfrog_steps,
                  log_stepsize_min, log_stepsize_max,
                  log_mass_min, log_mass_max,r1,s,a,
                  train_scale, initial_dist_mean).double()
    else:
        print("Initializing with const stepsize and mass")
        hmc = APMHMC(target, 2 * d, initial_dist,
                  num_hmc_steps, num_leapfrog_steps,
                  log_stepsize_const, log_stepsize_const,
                  log_mass_const, log_mass_const,r1,s,a,
                  train_scale, initial_dist_mean).double()

    hmc_optim = torch.optim.Adam(hmc.get_hmc_params(), lr=hmc_lr)
    if train_scale:
        print("Training with SKSD")
        scale_optim = torch.optim.Adam([hmc.scale], lr=hmc_scale_lr)
        g_optim = torch.optim.Adam([hmc.raw_g], lr=hmc_g_lr)
    else:
        print("Training without SKSD")

    losses = np.zeros((iters,))
    losses1 = np.zeros((iters,))

    log_step_sizes = np.zeros((iters, num_hmc_steps, 2 * d))
    log_masses = np.zeros((iters, num_hmc_steps, 2 * d))
    acc_probs = np.zeros((iters,))

    torch.save(hmc.state_dict(), 'pre_training_hmc_model.pt')

    print("\n Training HMC params")

    start_time = time.time()
    for i in tqdm(range(iters)):
        hmc_optim.zero_grad()

        samples, ap = hmc.forward(batch_size)
        samples0,_ = initial_dist.sample(batch_size)
        samples_gz0,_ = hmc.forward_z0(samples0, 1)

        w = target.log_prob(samples_gz0) - log_q(samples_gz0)
        w = torch.reshape(w, (batch_size,-1))
        w = torch.mean(w,1)

        elbo0 = torch.mean(target.log_prob(samples0)) + torch.log(2 * torch.tensor(np.pi))
        loss1 = -elbo0 - torch.mean(log_q(samples)) + torch.mean(w * log_q(samples0))
        loss = (b**2) / (-loss1) + loss1 / (b**2)

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
        losses1[i] = loss1.detach().numpy()

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
    total_time = time.time() - start_time
    print('losses1: ',losses1)
    print('losses: ',losses)
    print(hmc.state_dict())

    print('log_step_sizes: ',log_step_sizes)
    print('log_masses: ',log_masses)
    print('acc_probs: ',acc_probs)

    print(model_name + ' time: {}'.format(total_time))
    print(fin_time)

if __name__ == "__main__":
    main()
