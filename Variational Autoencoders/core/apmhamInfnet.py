import tensorflow as tf

from util.constant import log_2pi
from core.ham import apm_leapfrog,leapfrog
import numpy as np


class APMHamInfNetHEI:
    def __init__(self, num_lfsteps,
                 num_layers,
                 sample_dim,
                 r1,
                 s,
                 training=False,
                 min_step_size=0.01,
                 init_step_scale=1.0,
                 log_q0_std_init=0.0,
                 name_space="",
                 stop_gradient=True,
                 dtype=tf.float32):
        self.num_lfsteps = num_lfsteps
        self.num_layers = num_layers
        self.num_layers_max = num_layers
        self.stop_gradient = stop_gradient
        self.sample_dim = sample_dim
        self.dtype = dtype
        self.r1 = r1
        self.s = s


        self.lfstep_size_raw = tf.get_variable(name="{}lfstep_size".format(name_space),
                                               initializer=tf.constant(
                                                   init_step_scale * np.random.uniform(0.02, 0.05, size=(
                                                   num_layers, 1, sample_dim)),
                                                   dtype=dtype),
                                               trainable=True, dtype=dtype)

        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size

        self.raw_inflation = tf.get_variable(name="{}raw_inflation".format(name_space),
                                             initializer=tf.constant(1.0),
                                             trainable=True, dtype=dtype
                                             )

        self.inflation = tf.abs(self.raw_inflation)

    def __build_LF_graph_apmhmc(self, pot_fun, state_init, momemtum, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state, momemtum2: tf.less(layer_index, num_layers_effect)


        def _loopbody(layer_index, state, momemtum):
            state_new, momemtum_new = apm_leapfrog(x=state,
                                         v = momemtum,
                                         pot_fun=pot_fun,
                                         eps=self.lfstep_size[layer_index],
                                         r_var=1.0,
                                         r1 = self.r1,
                                         s = self.s,
                                         numleap=self.num_lfsteps,
                                         stop_gradient_pot=self.stop_gradient,
                                         back_prop=back_prop)
            return layer_index + 1, state_new, momemtum_new
        _, state_final,_ = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init, momemtum[0]))
        return state_final

    def __build_LF_graph_ksd(self, pot_fun, state_init, momemtum, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)

        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                         v=momemtum[layer_index],
                                         pot_fun=pot_fun,
                                         eps=tf.stop_gradient(self.lfstep_size[layer_index]),
                                         r_var=1.0,
                                         numleap=self.num_lfsteps,
                                         stop_gradient_pot=self.stop_gradient,
                                         back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final

    def build_elbo_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        """
        sample batch shape: sample_batch_size x input_data_batch_size x sample_dim
        potential batch shape: sample_batch_size x input_data_batch_size

        the list of shape of variables
        state_init: sample batch shape
        pot_energy_all_samples_final: potential batch shape
        log_q0_z: potential batch shape

        :param pot_fun: the potential function takes a batch of samples as input and outputs batch of potential values
        :param state_init_gen: take sample_batch_size and input_data_batch_size as input and outputs a batch of samples
        from initial distribution q_0 and their log probability log q_0
        :param sample_batch_size: the number of samples used to estimate the gradient
        :param input_data_batch_size: the batch size of input data, which must be compatible with potential function
        :param training: Boolean variable true for training / false for evaluation

        :return:
        elbo_mean: the Monte Carlo (sample average) estimation of loss function

        """
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size,
                                              tf.stop_gradient(self.inflation))

        momemtum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_final = self.__build_LF_graph_apmhmc(pot_fun, state_init, momemtum, back_prop=training)

        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood

        pot_gaussian_prior = 0.5 * tf.reduce_sum(state_final ** 2 + log_2pi, axis=-1)

        nelbo_per_sample = pot_energy_all_samples_final

        elbo_per_data = tf.reduce_mean(-nelbo_per_sample, axis=0)

        elbo_mean = tf.reduce_mean(elbo_per_data)

        recon_mean = tf.reduce_mean(pot_energy_all_samples_final - pot_gaussian_prior)

        return elbo_mean, recon_mean,state_init,state_final,elbo_per_data

    def build_ksd_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size, self.inflation)
        momemtum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)

        state_final = self.__build_LF_graph_ksd(pot_fun, state_init, momemtum, back_prop=training)

        def KSD_no_second_gradient(z, Sqx, flag_U=False):
            # dim_z is sample_size * latent_dim 
            # compute the rbf kernel
            K, dimZ = z.shape
            r = tf.reduce_sum(z * z, 1)
            # turn r into column vector
            r = tf.reshape(r, [-1, 1])
            pdist_square = r - 2 * tf.matmul(z, tf.transpose(z)) + tf.transpose(r)

            def get_median(v):
                v = tf.reshape(v, [-1])
                if v.get_shape()[0] % 2 == 1:
                    mid = v.get_shape()[0] // 2 + 1
                    return tf.nn.top_k(v, mid).values[-1]
                else:
                    mid1 = v.get_shape()[0] // 2
                    mid2 = v.get_shape()[0] // 2 + 1
                    return 0.5 * (tf.nn.top_k(v, mid1).values[-1] + tf.nn.top_k(v, mid2).values[-1])

            h_square = tf.stop_gradient(get_median(pdist_square))

            Kxy = tf.exp(- pdist_square / (2 * h_square))

            # now compute KSD
            Sqxdy = tf.matmul(tf.stop_gradient(Sqx), tf.transpose(z)) - \
                    tf.tile(tf.reduce_sum(tf.stop_gradient(Sqx) * z, 1, keepdims=True), (1, K))
            Sqxdy = -Sqxdy / h_square

            dxSqy = tf.transpose(Sqxdy)
            dxdy = -pdist_square / (h_square ** 2) + dimZ.value / h_square
            # M is a (K, K) tensor
            M = (tf.matmul(tf.stop_gradient(Sqx), tf.transpose(tf.stop_gradient(Sqx))) + \
                 Sqxdy + dxSqy + dxdy) * Kxy

            # the following for U-statistic
            if flag_U:
                M2 = M - tf.diag(tf.diag(M))
                return tf.reduce_sum(M2) / (K.value * (K.value - 1))

            # the following for V-statistic
            return tf.reduce_mean(M)

            # Now apply compute KSD for each input data in the mini-batch

        # pot_fun is neg-log-lik
        pot_energy_all_samples = pot_fun(state_final)  # sample_size * input_batch , neg log-lik
        grad_pot_all_samples = tf.gradients(ys=-pot_energy_all_samples, xs=state_final)[
            0]  # sample_size * input_batch* latent_dim

        cond = lambda batch_index, ksd_sum: tf.less(batch_index, input_data_batch_size)

        def _loopbody(batch_index, ksd_sum):
            return batch_index + 1, ksd_sum + KSD_no_second_gradient(state_final[:, batch_index, :],
                                                                     grad_pot_all_samples[:, batch_index, :])

        _, ksd_sum_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(
        1, KSD_no_second_gradient(state_final[:, 0, :], grad_pot_all_samples[:, 0, :])))
        return ksd_sum_final / input_data_batch_size

    def getParams(self):
        return self.lfstep_size_raw, self.raw_inflation

    def getInflation(self):
        return self.inflation

