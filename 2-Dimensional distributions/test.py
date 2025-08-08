
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import getval
from autograd.scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform

from targets import *

import tensorflow as tf
import tensorflow_probability as tfp

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


def init_random_params(L):
    eps = 0.01 + rs.rand(L, 2) * 0.015
    log_eps = np.log(eps)
    log_v_r = np.zeros([L, 2])
    s = 3 * np.ones((1, 2))
    mu0 = np.zeros((1, 2))
    log_sigma0 = np.zeros((1, 2))
    log_inflation = 0 * np.ones((1, 2))
    return np.concatenate((log_eps, log_v_r, s, mu0, log_sigma0, log_inflation), 0)


def leapfrog(z, r, eps, log_v_r, dlogP):
    for i in range(5):
        r_half = r - eps / 2.0 * -getval(dlogP(z))  # stops the gradient computation
        z = z + eps * r_half / np.exp(log_v_r)
        r = r_half - eps / 2.0 * -getval(dlogP(z))  # stops the gradient computation
    return z, r


def f(z, params):
    return logP(z) - log_q(z, params)


def log_q(z, params):
    k, zdim = z.shape
    mu0 = getval(params[-3, :])
    mu0 = np.ones((k, 1)) * mu0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)
    var = sigma0 ** 2
    var_inv = np.expand_dims(1 / var, 1) * np.eye(2)
    # np.expand_dims扩展维度，np.eye()单位矩阵
    det = np.prod(var)
    # np.prod默认为计算所有元素的乘积
    return np.log(1 / np.sqrt((2 * np.pi) ** zdim * det)) + np.diag(
        (-0.5 * np.matmul(np.matmul((z - mu0), var_inv), (z - mu0).T)))
    # np.diag 一维数组变为二维对角矩阵，二维矩阵输出对角线元素
    # np.matmul 矩阵相乘, .T 矩阵转置


def evaluate_objective(params):
    N = 100
    samples_HMC, samples0 = generate_samples_A(params, N)
    samples_gz0, acp = generate_samples_z0(params, samples0, n=10)

    sigma0 = np.exp(params[-2, :])
    var0 = sigma0 ** 2
    mu0 = params[-3, :]
    s1 = params[-4, 0]
    s2 = params[-4, 1]
    epsilon0 = rs.randn(N, params.shape[1])
    # samples0 = np.ones((N, 1)) * mu0 + epsilon0 * (np.ones((N, 1)) * sigma0)

    #w1 = getval(f(samples_gz0, params))  # [1000,]
    w = getval(logP(samples_gz0))
    w = np.reshape(w, (100, -1))  # [100,10]
    w = np.mean(w, 1)

    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-2, 0] + params[-2, 1]
    # loss1 = np.mean(logP(samples_HMC)) + elbo0
    loss1 = elbo0 + np.mean(log_q(getval(samples_HMC), params)) \
            - np.mean(w * log_q(getval(samples0), params))

    # loss_a0 = -np.mean(logP(samples_HMC)) - elbo0
    loss = (s1 ** 2) / (acp * loss1) - (acp * loss1) / (s2 ** 2)
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
    grad_objective = grad(evaluate_objective)
    epochs = 200

    # start = time.time()
    for epoch in range(epochs):
        # if epoch + 1 == 100:
        #    end = time.time()
        #    print("time: {}".format(end-start))
        t += 1
        print_perf(epoch, params)
        grad_params = grad_objective(params)
        m1 = beta1 * m1 + (1 - beta1) * grad_params
        m2 = beta2 * m2 + (1 - beta2) * grad_params ** 2
        m1_hat = m1 / (1 - beta1 ** t)
        m2_hat = m2 / (1 - beta2 ** t)

        params = params - alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)  # alpha is step size of adam

    return params


def generate_samples_HMC(params, n=100):
    log_eps = params[0: L, :]
    log_v_r = params[L: (2 * L), :]

    mu0 = getval(params[-3, :])
    mu0 = np.ones((n, 1)) * mu0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)

    log_inflation = getval(params[-1, :][0])
    inflation = np.exp(log_inflation)

    z = rs.randn(n, params.shape[1]) * (np.ones((n, 1)) * (sigma0 * inflation)) + mu0
    z0 = z

    for j in range(L):
        r = rs.randn(n, params.shape[1]) * np.exp(0.5 * log_v_r[j, :])
        z_new, r_new = leapfrog(z, r, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        p_acceptance = np.minimum(1,
                                  np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(r_new ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                         0.5 * np.sum(r ** 2 / np.exp(log_v_r[j, :]), 1)))
        accepted = rs.rand(n) < p_acceptance
        accepted = np.transpose(np.tile(accepted, (params.shape[1], 1)))
        z = z_new * accepted + (1 - accepted) * z

    return z, z0


def generate_samples_z0(params, z0, n=10):
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

    # z = rs.randn(n, params.shape[1]) * (np.ones((n, 1)) * (sigma0 * inflation)) + mu0
    acp = np.zeros(1000)
    for j in range(L):
        r = rs.randn(z.shape[0], params.shape[1]) * np.exp(0.5 * log_v_r[j, :])
        z_new, r_new = leapfrog(z, r, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        p_acceptance = np.minimum(1,
                                  np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(r_new ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                         0.5 * np.sum(r ** 2 / np.exp(log_v_r[j, :]), 1)))
        accepted = rs.rand(z.shape[0]) < p_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        acp += accepted
    mean_acp = np.mean(acp / L)
    return z, mean_acp


def generate_samples_pmmc(params, n=100):
    s = 0.4
    log_eps = params[0: L, :]
    log_v_r = params[L: (2 * L), :]

    mu0 = getval(params[-3, :])
    mu0 = np.ones((n, 1)) * mu0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)

    log_inflation = getval(params[-1, :][0])
    inflation = np.exp(log_inflation)

    z = rs.randn(n, params.shape[1]) * (np.ones((n, 1)) * (sigma0 * inflation)) + mu0
    # z = 随机[100,params(列数)]*(全1[100,1]*sigma0[2,])[100,2] + mu0[100,2]
    r0 = rs.randn(n, params.shape[1]) * np.exp(0.5 * log_v_r[0, :])  # [100,2]
    z0 = z

    for j in range(L):
        u = rs.randn(n, params.shape[1]) * np.exp(0.5 * log_v_r[j, :])  # [100,2]
        r2 = np.sqrt(1 - s) * r0 + np.sqrt(s) * u
        u2 = -np.sqrt(s) * r0 + np.sqrt(1 - s) * u
        # r_acceptance = np.minimum(1, np.exp(- 0.5 * np.sum(r2 ** 2 / np.exp(0), 1) + \
        #                                     0.5 * np.sum(r0 ** 2 / np.exp(0), 1) - \
        #                                     0.5 * np.sum(u2 ** 2 / np.exp(0), 1) + \
        #                                     0.5 * np.sum(u ** 2 / np.exp(0), 1)))
        r_acceptance = np.minimum(1, np.exp(- 0.5 * np.sum(r2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(r0 ** 2 / np.exp(log_v_r[j, :]), 1) - \
                                            0.5 * np.sum(u2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(u ** 2 / np.exp(log_v_r[j, :]), 1)))
        r_accepted = rs.rand(n) < r_acceptance
        r_accepted_tile = np.transpose(np.tile(r_accepted, (params.shape[1], 1)))  # 把accepted[1，100]变为[100,2]
        r = r2 * r_accepted_tile + (1 - r_accepted_tile) * r0

        z_new, r0 = leapfrog(z, r, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(r0 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(r ** 2 / np.exp(log_v_r[j, :]), 1)))  #

        accepted = rs.rand(n) < p_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))  # 把accepted[1，100]变为[100,2]
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        r0 = r0 * accepted_tile - (1 - accepted_tile) * r

    return z, z0  # 返回最终位置z(30次leapfrog后的位置)，初始位置z0，mean_acp平均接受-


def generate_samples_pmmc2(params, n=100):  # 生成样本
    log_eps = params[0: L, :]
    log_v_r = params[L: (2 * L), :]

    mu0 = getval(params[-3, :])
    mu0 = np.ones((n, 1)) * mu0
    log_sigma0 = getval(params[-2, :])
    sigma0 = np.exp(log_sigma0)

    log_inflation = getval(params[-1, :][0])
    inflation = np.exp(log_inflation)

    z = rs.randn(n, params.shape[1]) * (np.ones((n, 1)) * sigma0) + mu0
    z0 = z
    # z = 随机[100,params(列数)]*(全1[100,1]*sigma0[2,])[100,2] + mu0[100,2]
    r0 = rs.randn(n, params.shape[1]) * np.exp(0.5 * log_v_r[0, :])  # [100,2]
    s = np.ones((n, 2)) * 0.5  # 噪声
    for j in range(L):
        u = rs.randn(n, params.shape[1]) * np.exp(0.5 * log_v_r[j, :])  # [100,2]
        r2 = np.sqrt(1 - s) * r0 + np.sqrt(s) * u
        u2 = -np.sqrt(s) * r0 + np.sqrt(1 - s) * u

        r_acceptance = np.minimum(1, np.exp(- 0.5 * np.sum(r2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(r0 ** 2 / np.exp(log_v_r[j, :]), 1) - \
                                            0.5 * np.sum(u2 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(u ** 2 / np.exp(log_v_r[j, :]), 1)))
        r_accepted = rs.rand(n) < r_acceptance
        r_accepted_tile = np.transpose(np.tile(r_accepted, (params.shape[1], 1)))  # 把accepted[1，100]变为[100,2]
        sr = np.transpose(np.tile(r_acceptance, (params.shape[1], 1)))

        r = r2 * r_accepted_tile + (1 - r_accepted_tile) * r0

        z_new, r0 = leapfrog(z, r, np.exp(log_eps[j, :]), log_v_r[j, :], dlogP)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) - 0.5 * np.sum(r0 ** 2 / np.exp(log_v_r[j, :]), 1) + \
                                            0.5 * np.sum(r ** 2 / np.exp(log_v_r[j, :]), 1)))

        s = np.transpose(np.tile(p_acceptance, (params.shape[1], 1)))
        # s = 1-s
        s = np.ones((n, 2)) * 0.1 + 0.9 * (0.9 - s)
        accepted = rs.rand(n) < p_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))  # 把accepted[1，100]变为[100,2]
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        r0 = r0 * accepted_tile - (1 - accepted_tile) * r

    return z,z0


if __name__ == "__main__":
    fun = 6
    generate_samples_A = generate_samples_pmmc2
    generate_samples_S = generate_samples_pmmc2

    logP = logP_gauss
    dlogP = dlogP_gauss

    L = 30
    rs = npr.RandomState(0)
        # print(rs.randn(1))
    params = init_random_params(L)
    params = adam(evaluate_objective, params)
    z, _ = generate_samples_pmmc2(params, n=100000)
    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))

        # exp_params = np.exp(params)
        # exp_params[-3, :] = np.log(exp_params[-3, :])
        # exp_params[-1,1] = None
        # print("step_sizes, mu, sigma and inflation: {}".format(exp_params))

        # ess= tfp.mcmc.effective_sample_size(z)

    # z1=z[:,0]
    # z2=z[:,1]
    # plt.hist2d(z1, z2, bins=(300, 300))
    # #plt.xlim(-4,4)
    # #plt.ylim(-4,4)
    # plt.show()
