from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import getval
from autograd.scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform

from targets import *
from datetime import datetime

# import tensorflow as tf
# import tensorflow_probability as tfp

import time


def KSD(z, Sqx, flag_U=False):
    # compute the rbf kernel
    K, dimZ = z.shape
    sq_dist = pdist(z)
    pdist_square = squareform(sq_dist) ** 2

    h_square = 1.0

    Kxy = np.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = np.dot(getval(Sqx), z.T) - np.tile(np.sum(getval(Sqx) * z, 1, keepdims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = Sqxdy.T
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square

    # M is a (K, K) tensor
    M = (np.dot(getval(Sqx), getval(Sqx).T) + Sqxdy + dxSqy + dxdy) * Kxy

    # U-statistic
    if flag_U:
        M2 = M - np.diag(np.diag(M))
        return np.sum(M2) / (K * (K - 1))
    # V-statistic
    else:
        return np.sum(M) / (K * K)

def init_random_params(L, b):
    eps = 0.01 + rs.rand(L, 2) * 0.015
    log_eps = np.log(eps)
    log_v_r = np.zeros([L, 2])
    b = b * np.ones((1, 2))
    m0 = np.zeros((1, 2))
    log_sigma0 = np.zeros((1, 2))
    log_inflation = np.zeros((1, 2))
    return np.concatenate((log_eps, log_v_r, b, m0, log_sigma0, log_inflation), 0)

def leapfrog(z, v, eps, log_v_r, dlogP):
    for i in range(5):
        v_half = v - eps / 2.0 * -getval(dlogP(z))  # stops the gradient computation
        z = z + eps * v_half / np.exp(log_v_r)
        v = v_half - eps / 2.0 * -getval(dlogP(z))  # stops the gradient computation
    return z, v


def generate_samples_apmhmc_z0(params, z0, n=10):
    z = z0
    z = np.tile(z, (1, n)).reshape((-1, 2))
    log_eps = params[0: L, :]
    log_v_r = params[L: (2 * L), :]

    mu0 = getval(params[-3, :])
    mu0 = np.ones((n, 1)) * mu0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)

    log_inflation = getval(params[-1, :][0])
    inflation = np.exp(log_inflation)

    acp = np.zeros(z.shape[0])
    v0 = rs.randn(z.shape[0], params.shape[1]) * np.exp(0.5 * log_v_r[0, :])
    r = np.zeros((z.shape[0], 1))
    for j in range(L):
        mu = rs.randn(z.shape[0], params.shape[1]) * np.exp(0.5 * log_v_r[j, :])
        v2 = np.sqrt(1 - r) * v0 + np.sqrt(r) * mu
        mu2 = -np.sqrt(r) * v0 + np.sqrt(1 - r) * mu
        v_acceptance = np.minimum(1, np.exp(- 0.5 * np.sum(v2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(v0 ** 2 / np.exp(log_v_r[j, :]), 1) - \
                                            0.5 * np.sum(mu2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(mu ** 2 / np.exp(log_v_r[j, :]), 1)))
        v_accepted = rs.rand(r.shape[0]) < v_acceptance
        v_accepted_tile = np.transpose(np.tile(v_accepted, (params.shape[1], 1)))
        v0 = v2 * v_accepted_tile - (1 - v_accepted_tile) * v0

        z_new, v_new = leapfrog(z, v0, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        z_acceptance = np.minimum(1,
                                  np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(v_new ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                         0.5 * np.sum(v0 ** 2 / np.exp(log_v_r[j, :]), 1)))
        accepted = rs.rand(z.shape[0]) < z_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))

        z = z_new * accepted_tile + (1 - accepted_tile) * z
        v0 = v_new * accepted_tile - (1 - accepted_tile) * v0
        z_acp = np.transpose(np.tile(z_acceptance, (params.shape[1], 1)))

        r = np.ones((z.shape[0], 2)) * 0.1 + 0.9 * (0.9 - z_acp)
        acp += accepted
    mean_acp = np.mean(acp / L)
    return z, mean_acp

def generate_samples_apmhmc(params, z0=np.array([]), n=100, m=10):
    log_eps = params[0: L, :]
    log_v_r = params[L: (2 * L), :]

    m0 = getval(params[-3, :])
    m0 = np.ones((n, 1)) * m0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)

    log_inflation = getval(params[-1, :][0])
    inflation = np.exp(log_inflation)
    if z0.size == 0:
        z0 = rs.randn(n, params.shape[1]) * (np.ones((n, 1)) * (sigma0 * inflation)) + m0
        z = z0
    else:
        z = np.tile(z0, (1, m)).reshape((-1, 2))

    v0 = rs.randn(z.shape[0], params.shape[1]) * np.exp(0.5 * log_v_r[0, :])
    r = np.zeros((n, 2))  

    for j in range(L):
        mu = rs.randn(z.shape[0], params.shape[1]) * np.exp(0.5 * log_v_r[j, :])
        v2 = np.sqrt(1 - r) * v0 + np.sqrt(r) * mu
        mu2 = -np.sqrt(r) * v0 + np.sqrt(1 - r) * mu
        v_acceptance = np.minimum(1, np.exp(- 0.5 * np.sum(v2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(v0 ** 2 / np.exp(log_v_r[j, :]), 1) - \
                                            0.5 * np.sum(mu2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(mu ** 2 / np.exp(log_v_r[j, :]), 1)))
        v_accepted = rs.rand(r.shape[0]) < v_acceptance
        v_accepted_tile = np.transpose(np.tile(v_accepted, (params.shape[1], 1)))
        v0 = v2 * v_accepted_tile - (1 - v_accepted_tile) * v0

        z_new, v_new = leapfrog(z, v0, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        p_acceptance = np.minimum(1,
                                  np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(v_new ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                         0.5 * np.sum(v0 ** 2 / np.exp(log_v_r[j, :]), 1)))
        accepted = rs.rand(z.shape[0]) < p_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))

        z = z_new * accepted_tile + (1 - accepted_tile) * z
        v0 = v_new * accepted_tile - (1 - accepted_tile) * v0
        acp = np.transpose(np.tile(p_acceptance, (params.shape[1], 1)))

        r = np.ones((n, 2)) * 0.1 + 0.9 * (0.9 - acp)

    return z, z0

def f(z, params):
    return logP(z) - log_q(z, params)

def log_q(z, params):
    k, zdim = z.shape
    m0 = getval(params[-3, :])
    m0 = np.ones((k, 1)) * m0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)
    var = sigma0 ** 2
    var_inv = np.expand_dims(1 / var, 1) * np.eye(2)
    det = np.prod(var)
    return np.log(1 / np.sqrt((2 * np.pi) ** zdim * det)) + np.diag(
        (-0.5 * np.matmul(np.matmul((z - m0), var_inv), (z - m0).T)))

def evaluate_objective(params):
    N = 100
    samples, samples0 = generate_samples_apmhmc(params, n=100)
    samples_gz0, acp = generate_samples_apmhmc_z0(params, samples0, n=10)

    sigma0 = np.exp(params[-2, :])
    var0 = sigma0 ** 2
    m0 = params[-3, :]
    b1 = params[-4, 0]
    b2 = params[-4, 1]

    w = getval(f(samples_gz0, params))
    w = np.reshape(w, (100, -1))  
    w = np.mean(w, 1)

    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-2, 0] + params[-2, 1]
    vcd = -elbo0 - np.mean(log_q(getval(samples), params)) \
            + np.mean(w * log_q(getval(samples0), params))

    loss = (b1 ** 2) / (-vcd) + vcd / (b2 ** 2)
    return loss

def adam(evaluate_objective, params):
    print("    Step       |     objective      ")

    def print_perf(epoch, params):
        objective = evaluate_objective(params)
        print("{0:15}|{1:15}".format(epoch, -objective))

    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.05
    t = 0
    epochs = 200

    grad_objective = grad(evaluate_objective)

    for epoch in range(epochs):
        t += 1
        print_perf(epoch, params)
        grad_params = grad_objective(params)
        m1 = beta1 * m1 + (1 - beta1) * grad_params
        m2 = beta2 * m2 + (1 - beta2) * grad_params ** 2
        m1_hat = m1 / (1 - beta1 ** t)
        m2_hat = m2 / (1 - beta2 ** t)

        params = params - alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)


    return params




if __name__ == "__main__":
    rs = npr.RandomState(0)
    L = 30
    b = 3
    logP = logP_gauss
    dlogP = dlogP_gauss

    params = init_random_params(L,b)
    params = adam(evaluate_objective, params)

    z, _ = generate_samples_apmhmc(params, n=100000)
    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))

    z1=z[:,0]
    z2=z[:,1]
    plt.hist2d(z1, z2, bins=(300, 300))
    #plt.xlim(-4,4)
    #plt.ylim(-4,4)
    plt.show()
