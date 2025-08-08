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
        s = np.ones((n, 2)) * 0.5 + 0.7 * (0.7 - s)
        accepted = rs.rand(n) < p_acceptance
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[1], 1)))  # 把accepted[1，100]变为[100,2]
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        r0 = r0 * accepted_tile - (1 - accepted_tile) * r

    return z,z0