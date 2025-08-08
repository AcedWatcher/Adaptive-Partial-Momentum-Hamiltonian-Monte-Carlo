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
from core.utils import calculate_log_likelihood
from core.models import HMC

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.chdir(path)
    else:
        print('File already exists')
        os.chdir(path)

#@hydra.main(config_path='conf', config_name="evaluate_model")
def main():
    path = ''
    data_path = path + ''
    initial_name = ''
    model_name = ''

    initial_dist_path = path + '/InitialDist_' + initial_name + ''
    hmc_path = path + '/' + model_name + ''

    evaluate_name = '-hmc'
    mkdir(path + r'/EvaluateModels/'+ model_name + evaluate_name)
    hmc_model_includes_scale = False
    test_initial_dist_model = False # T：evaluate initial，F：evaluate model

    d = 64
    sigma_0 = 0.005
    tau = 0.01
    num_hmc_steps = 20
    num_leapfrog_steps = 5

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    is_test = True
    is_validation = False
    num_w_samples = 10000
    num_repeats = 20

    test_hmc_model = True


    print("Evaluating model on {} data".format('validation' if is_validation else 'test'))

    print("Loading observed data from", data_path)
    X_train, y_train, \
        X_validation, y_validation, \
        X_test, y_test, true_w, true_lamb = \
        np.load(data_path + 'X_train.npy'), \
            np.load(data_path + 'y_train.npy'), \
            np.load(data_path + 'X_validation.npy'), \
            np.load(data_path + 'y_validation.npy'), \
            np.load(data_path + 'X_test.npy'), \
            np.load(data_path + 'y_test.npy'), \
            np.load(data_path + 'w.npy'), \
            np.load(data_path + 'lambda.npy')

    assert is_test != is_validation
    if is_validation:
        X_star = X_validation
        y_star = y_validation
    elif is_test:
        X_star = X_test
        y_star = y_test

    X_train, y_train, true_w, true_lamb = \
            torch.from_numpy(X_train).double(), \
            torch.from_numpy(y_train).double(), \
            torch.from_numpy(true_w).double(), \
            torch.from_numpy(true_lamb).double(),\

    target = HorseshoeTarget(X_train, y_train, tau, sigma_0)

    if test_initial_dist_model:
        print("Loading an initial dist model")

        initial_dist = nf.core.NormalizingFlow(
            q0=nf.distributions.DiagGaussian(
                shape=2 * d
            ),
            flows=None,
            p=target
        ).double()
        print("Loading initial dist from", initial_dist_path)
        initial_dist.load_state_dict(torch.load(initial_dist_path))

        print("Sampling initial distribution")
        generate_sample = initial_dist.sample

    else:
        print("Loading a HMC model")
        initial_dist = nf.core.NormalizingFlow(
            q0=nf.distributions.DiagGaussian(
                shape=2 * d
            ),
            flows=None,
            p=target
        ).double()

        hmc = HMC(target, 2 * d, initial_dist,
                  num_hmc_steps, num_leapfrog_steps,
                  0, 1, 0, 1, hmc_model_includes_scale,
                  torch.zeros(2 * d)).double()

        print("Loading HMC model from", hmc_path)
        hmc.load_state_dict(torch.load(hmc_path))

        print("Sampling hmc model")
        generate_sample = hmc.forward

    log_likelihood_vals = np.zeros((num_repeats))

    print("Calculating log likelihood values")
    start_time = time.time()
    for i in tqdm(range(num_repeats)):
        samples, _ = generate_sample(num_w_samples)
        samples = samples.detach().cpu().numpy()
        samples = samples[:, 0:d]

        sub_samples = samples
        log_likelihood = calculate_log_likelihood(
            sub_samples, X_star, y_star, sigma_0)
        log_likelihood_vals[i] = log_likelihood

    np.save('log_likelihood_vals.npy', log_likelihood_vals)
    np.save('w_samples.npy', samples)
    print('time: {}s'.format(time.time()-start_time))
    print('log_likelihood_vals:', log_likelihood_vals)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(model_name + evaluate_name + ', log_mean:', np.mean(log_likelihood_vals))
    #print('log_likelihood_mean:', np.mean(log_likelihood_vals))


if __name__ == "__main__":
    main()
