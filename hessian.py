import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian as jac
import progressbar
tod = torch.distributions
from tensorflow.keras.utils import Progbar
import pkbar
from torch.distributions import Categorical


class Condpf(torch.nn.Module):

    def __init__(self, model, param):
        super().__init__()
        self.model = model
        self.mu = model.mu
        self.sigma = model.sigma
        self.llg = model.likelihood_logscale
        self.l = param[0]
        self.T = param[1]
        self.N = param[2]
        self.dx = param[3]
        self.dy = param[4]
        self.initial_val = param[5]

    # output shape (2**l, N, dx), input shape  (N, dx)
    def unit_path_update(self, x):
        hl = 2 ** (-self.l)
        x_out = torch.zeros(int(2 ** (self.l) + 1), x.shape[0], x.shape[-1])
        x_out[0] = x
        for dt in range(2 ** self.l):
            dw = torch.randn(x.shape[0], self.dx, 1) * np.sqrt(hl)
            x_out[dt + 1] = x_out[dt] + self.mu(x_out[dt]) * hl + (self.sigma(x_out[dt]) @ dw)[..., 0]
        return x_out[1:]

    # Identital paths update
    def driving_update(self, x, x1):
        hl = 2 ** (-self.l)
        x_out = torch.zeros(int(2 ** (self.l) + 1), x.shape[0], x.shape[-1])
        x1_out = torch.zeros(int(2 ** (self.l) + 1), x1.shape[0], x1.shape[-1])
        x_out[0] = x
        x1_out[0] = x1

        for dt in range(2 ** self.l):
            dw = torch.randn(x.shape[0], self.dx, 1) * np.sqrt(hl)
            x_out[dt + 1] = x_out[dt] + self.mu(x_out[dt]) * hl + (self.sigma(x_out[dt]) @ dw)[..., 0]
            x1_out[dt + 1] = x1_out[dt] + self.mu(x1_out[dt]) * hl + (self.sigma(x1_out[dt]) @ dw)[..., 0]

        return x_out[1:], x1_out[1:]

    # Coupled finer & coarser update in unit time, (2**l, N, dx) (2**(l-1), N, dx)
    def coupled_update(self, x1, x2):
        hl = 2 ** (-self.l)
        hlm1 = 2 ** (-self.l + 1)
        x1_out = torch.zeros(int(2 ** (self.l) + 1), x1.shape[0], x1.shape[-1])
        x2_out = torch.zeros(int(2 ** (self.l - 1) + 1), x2.shape[0], x2.shape[-1])
        x1_out[0] = x1
        x2_out[0] = x2

        for dt1 in range(2 ** (self.l - 1)):
            dw1 = torch.randn(x1.shape[0], self.dx, 1) * np.sqrt(hl)
            dw2 = torch.randn(x2.shape[0], self.dx, 1) * np.sqrt(hl)
            dw = dw1 + dw2
            x1_out[2 * dt1 + 1] = x1_out[2 * dt1] + self.mu(x1_out[2 * dt1]) * hl + (self.sigma(x1_out[2 * dt1]) @ dw1)[
                ..., 0]
            x1_out[2 * dt1 + 2] = x1_out[2 * dt1 + 1] + self.mu(x1_out[2 * dt1 + 1]) * hl + \
                                  (self.sigma(x1_out[2 * dt1 + 1]) @ dw2)[..., 0]
            x2_out[dt1 + 1] = x2_out[dt1] + self.mu(x2_out[dt1]) * hlm1 + (self.sigma(x2_out[dt1]) @ dw)[..., 0]

        return x1_out[1:], x2_out[1:]

    # initial path generation, output shape (T*2**l+1, dx)
    def initial_path_gen(self):
        un = torch.zeros(self.getind(self.T) + 1, 1, self.dx) + self.initial_val
        for t in range(self.T):
            start_ind = self.getind(t)
            update_ind = self.getind(t + 1)
            un[start_ind + 1:update_ind + 1] = self.unit_path_update(un[start_ind])
        return torch.squeeze(un)

    # Resampling input multi-dimensional particle x
    def resampling(self, weight, gn, x):
        N = self.N
        ess = 1 / ((weight ** 2).sum())
        if ess <= (N / 2):
            ## Sample with uniform dice
            dice = np.random.random_sample(N)
            ## np.cumsum obtains CDF out of PMF
            bins = np.cumsum(weight)
            bins[-1] = np.max([1, bins[-1]])
            ## np.digitize gets the indice of the bins where the dice belongs to
            x_hat = x[:, np.digitize(dice, bins), :]
            ## after resampling we reset the accumulating weight
            gn = torch.zeros(N)
        if ess > (N / 2):
            x_hat = x

        return x_hat, gn

    # Resampling input multi-dimensional particle x
    def pure_resampling(self, weight, gn, x):
        N = self.N
        ## Sample with uniform dice
        dice = np.random.random_sample(N)
        ## np.cumsum obtains CDF out of PMF
        bins = np.cumsum(weight)
        bins[-1] = np.max([1, bins[-1]])
        ## np.digitize gets the indice of the bins where the dice belongs to
        x_hat = x[:, np.digitize(dice, bins), :]
        ## after resampling we reset the accumulating weight
        gn = torch.zeros(N)

        return x_hat, gn

    # Sampling out according to the weight
    def sample_output(self, weight, x):
        ## Sample with uniform dice
        dice = np.random.random_sample(1)
        ## np.cumsum obtains CDF out of PMF
        bins = np.cumsum(weight)
        bins[-1] = np.max([1, bins[-1]])
        ## np.digitize gets the indice of the bins where the dice belongs to
        x_hat = x[:, np.digitize(dice, bins), :]
        ## return the sampled particle path
        return torch.squeeze(x_hat)

    def getind(self, t):
        return int(2 ** (self.l) * t)

    def getcind(self, t):
        return int(2 ** (self.l - 1) * t)

    # input_path of shape (2**l*T+1, dx)
    def condpf_kernel(self, input_path, observe_path):

        un = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un_hat = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        gn = torch.zeros(self.N)

        for t in range(self.T):
            start_ind = self.getind(t)
            un[:start_ind + 1] = un_hat[:start_ind + 1]
            # Euler update
            update_ind = self.getind(t + 1)
            un[start_ind + 1:update_ind + 1] = self.unit_path_update(un[start_ind])
            # Main point for conditional PF is that the last particle is fixed, and it joins the resampling process
            un[:, -1] = input_path

            # Cumulating weight function
            gn = self.llg(un[update_ind], observe_path[t + 1]) + gn
            what = torch.exp(gn - torch.max(gn))
            wn = what / torch.sum(what)
            wn = wn.detach().numpy()

            # Resampling
            un_hat[:update_ind + 1], gn = self.resampling(wn, gn, un[:update_ind + 1])
            un_hat[:, -1] = input_path

        # Sample out a path and output it
        return self.sample_output(wn, un)

    # Markov chain generation with CondPF, initial chain generated with built-in function
    def chain_gen_condpf(self, num_step, observe_path):
        x_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx) + self.initial_val
        x_chain[0] = self.initial_path_gen()
        for step in range(num_step):
            x_chain[step + 1] = self.condpf_kernel(x_chain[step], observe_path)
        return x_chain

    # Driving CCPF, both paths has same discretization levels and uses same BM in update
    def drive_ccpf_kernel(self, input_path1, input_path2, observe_path):

        un1 = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un1_hat = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un2 = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un2_hat = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val

        gn1 = torch.zeros(self.N)
        gn2 = torch.zeros(self.N)

        for t in range(self.T):
            start_ind1 = self.getind(t)
            start_ind2 = self.getind(t)

            un1[:start_ind1 + 1] = un1_hat[:start_ind1 + 1]
            un2[:start_ind2 + 1] = un2_hat[:start_ind2 + 1]
            # Euler update
            update_ind1 = self.getind(t + 1)
            update_ind2 = self.getind(t + 1)

            un1[start_ind1 + 1:update_ind1 + 1], un2[start_ind2 + 1:update_ind2 + 1] = self.driving_update(
                un1[start_ind1], un2[start_ind2])
            # Main point for conditional PF is that the last particle is fixed, and it joins the resampling process
            un1[:, -1] = input_path1
            un2[:, -1] = input_path2

            # Cumulating weight function
            gn1 = self.llg(un1[update_ind1], observe_path[t + 1]) + gn1
            what1 = torch.exp(gn1 - torch.max(gn1))
            wn1 = what1 / torch.sum(what1)
            wn1 = wn1.detach().numpy()

            gn2 = self.llg(un2[update_ind2], observe_path[t + 1]) + gn2
            what2 = torch.exp(gn2 - torch.max(gn2))
            wn2 = what2 / torch.sum(what2)
            wn2 = wn2.detach().numpy()

            # Resampling
            un1_hat[:update_ind1 + 1], gn1, un2_hat[:update_ind2 + 1], gn2 = self.coupled_maximal_resampling(wn1, wn2,
                                                                                                             gn1, gn2,
                                                                                                             un1[
                                                                                                             :update_ind1 + 1],
                                                                                                             un2[
                                                                                                             :update_ind2 + 1])
            un1_hat[:, -1] = input_path1
            un2_hat[:, -1] = input_path2

        # Sample out a path and output it
        path1_output, path2_output = self.coupled_maximal_sample(wn1, wn2, un1, un2, 1)
        return path1_output[:, 0, :], path2_output[:, 0, :]

    def coupled_maximal_sample(self, weight1, weight2, x1, x2, N):
        # Initialize
        x1_hat = torch.zeros(x1.shape[0], N, self.dx)
        x2_hat = torch.zeros(x2.shape[0], N, self.dx)

        # Calculating many weights
        unormal_min_weight = np.minimum(weight1, weight2)
        min_weight_sum = np.sum(unormal_min_weight)
        min_weight = unormal_min_weight / min_weight_sum
        unormal_reduce_weight1 = weight1 - unormal_min_weight
        unormal_reduce_weight2 = weight2 - unormal_min_weight

        ## Sample with uniform dice
        dice = np.random.random_sample(N)
        ## [0] takes out the numpy array which is suitable afterwards
        coupled = np.where(dice <= min_weight_sum)[0]
        independ = np.where(dice > min_weight_sum)[0]
        ncoupled = np.sum(dice <= min_weight_sum)
        nindepend = np.sum(dice > min_weight_sum)

        if ncoupled >= 0:
            dice1 = np.random.random_sample(ncoupled)
            bins = np.cumsum(min_weight)
            bins[-1] = np.max([1, bins[-1]])
            x1_hat[:, coupled, :] = x1[:, np.digitize(dice1, bins), :]
            x2_hat[:, coupled, :] = x2[:, np.digitize(dice1, bins), :]

        ## nindepend>0 implies min_weight_sum>0 imples np.sum(unormal_reduce_weight*) is positive, thus the division won't report error
        if nindepend > 0:
            reduce_weight1 = unormal_reduce_weight1 / np.sum(unormal_reduce_weight1)
            reduce_weight2 = unormal_reduce_weight2 / np.sum(unormal_reduce_weight2)
            dice2 = np.random.random_sample(nindepend)
            bins1 = np.cumsum(reduce_weight1)
            bins1[-1] = np.max([1, bins1[-1]])
            bins2 = np.cumsum(reduce_weight2)
            bins2[-1] = np.max([1, bins2[-1]])
            x1_hat[:, independ, :] = x1[:, np.digitize(dice2, bins1), :]
            x2_hat[:, independ, :] = x2[:, np.digitize(dice2, bins2), :]

        return x1_hat, x2_hat

    def coupled_maximal_resampling(self, weight1, weight2, gn1, gn2, x1, x2):
        ess = 1 / ((weight1 ** 2).sum())
        if ess <= (self.N / 2):
            # When resampling happens, unormalized likelihood function reset
            gn1 = torch.zeros(self.N)
            gn2 = torch.zeros(self.N)
            # Maimal coupled sampling
            x1_hat, x2_hat = self.coupled_maximal_sample(weight1, weight2, x1, x2, self.N)
        if ess > (self.N / 2):
            x1_hat, x2_hat = x1, x2

        return x1_hat, gn1, x2_hat, gn2

    def initial_lag_2path(self, observe_path):
        hl = 2 ** (-self.l)
        time_len = self.getind(self.T)

        ## Initial value
        un1 = torch.zeros(time_len + 1, 1, self.dx) + self.initial_val
        un2 = torch.zeros(time_len + 1, 1, self.dx) + self.initial_val

        ## Coupled Propagation
        for t in range(self.T):
            start_ind = self.getind(t)
            # Euler update
            update_ind = self.getind(t + 1)
            un1[start_ind + 1:update_ind + 1] = self.unit_path_update(un1[start_ind])
            un2[start_ind + 1:update_ind + 1] = self.unit_path_update(un2[start_ind])

        ## Lag-one forward for the first path
        un1_lag_one_forward = self.condpf_kernel(un1[:, 0, :], observe_path)
        return un1_lag_one_forward, un2[:, 0, :]

    # generate a chain of coupled particles with Driving CCPF of length 'num_step+1', including the starting position
    # both paths use same BM in update
    def chain_gen_dccpf(self, num_step, observe_path):

        x1_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx)
        x2_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx)
        x1_chain[0], x2_chain[0] = self.initial_lag_2path(observe_path)

        for step in range(num_step):
            x1_chain[step + 1], x2_chain[step + 1] = self.drive_ccpf_kernel(x1_chain[step], x2_chain[step],
                                                                            observe_path)
        return x1_chain, x2_chain

    # function for test
    def any_coupled_2path(self):
        hl = 2 ** (-self.l)
        time_len1 = self.getind(self.T)
        time_len2 = self.getcind(self.T)

        ## Initial value
        un1 = torch.randn(time_len1 + 1, self.dx) + self.initial_val
        un2 = torch.randn(time_len2 + 1, self.dx) + self.initial_val
        return un1, un2

    # function for test
    def chain_gen_ccpf(self, num_step, observe_path):
        x1_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx)
        x2_chain = torch.zeros(num_step + 1, self.getcind(self.T) + 1, self.dx)
        x1_chain[0], x2_chain[0] = self.any_coupled_2path()

        for step in range(num_step):
            x1_chain[step + 1], x2_chain[step + 1] = self.ccpf_kernel(x1_chain[step], x2_chain[step], observe_path)
        return x1_chain, x2_chain

    # Coupled Conditional Particl Filter Markov Kernel, two path are coupled path of level l and l-1
    def ccpf_kernel(self, input_path1, input_path2, observe_path):

        un1 = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un1_hat = torch.zeros(self.getind(self.T) + 1, self.N, self.dx) + self.initial_val
        un2 = torch.zeros(self.getcind(self.T) + 1, self.N, self.dx) + self.initial_val
        un2_hat = torch.zeros(self.getcind(self.T) + 1, self.N, self.dx) + self.initial_val
        gn1 = torch.zeros(self.N)
        gn2 = torch.zeros(self.N)

        for t in range(self.T):
            start_ind1 = self.getind(t)
            start_ind2 = self.getcind(t)

            un1[:start_ind1 + 1] = un1_hat[:start_ind1 + 1]
            un2[:start_ind2 + 1] = un2_hat[:start_ind2 + 1]
            # Euler update
            update_ind1 = self.getind(t + 1)
            update_ind2 = self.getcind(t + 1)

            un1[start_ind1 + 1:update_ind1 + 1], un2[start_ind2 + 1:update_ind2 + 1] = self.coupled_update(
                un1[start_ind1], un2[start_ind2])
            # Main point for conditional PF is that the last particle is fixed, and it joins the resampling process
            un1[:, -1] = input_path1
            un2[:, -1] = input_path2

            # Cumulating weight function
            gn1 = self.llg(un1[update_ind1], observe_path[t + 1]) + gn1
            what1 = torch.exp(gn1 - torch.max(gn1))
            wn1 = what1 / torch.sum(what1)
            wn1 = wn1.detach().numpy()

            gn2 = self.llg(un2[update_ind2], observe_path[t + 1]) + gn2
            what2 = torch.exp(gn2 - torch.max(gn2))
            wn2 = what2 / torch.sum(what2)
            wn2 = wn2.detach().numpy()

            # Resampling
            un1_hat[:update_ind1 + 1], gn1, un2_hat[:update_ind2 + 1], gn2 = self.coupled_maximal_resampling(wn1, wn2,
                                                                                                             gn1, gn2,
                                                                                                             un1[
                                                                                                             :update_ind1 + 1],
                                                                                                             un2[
                                                                                                             :update_ind2 + 1])
            un1_hat[:, -1] = input_path1
            un2_hat[:, -1] = input_path2

        # Sample out a path and output it
        path1_output, path2_output = self.coupled_maximal_sample(wn1, wn2, un1, un2, 1)
        return path1_output[:, 0, :], path2_output[:, 0, :]

    # Generated two coupled particle paths, one of them is lagged-one forward through Coupled Conditional Particle Filter
    def initial_lag_4path(self, observe_path):
        hl = 2 ** (-self.l)
        time_len = self.getind(self.T)
        time_len1 = self.getcind(self.T)

        ## Initial value
        un1 = torch.zeros(time_len + 1, 1, self.dx) + self.initial_val
        un2 = torch.zeros(time_len + 1, 1, self.dx) + self.initial_val
        cn1 = torch.zeros(time_len1 + 1, 1, self.dx) + self.initial_val
        cn2 = torch.zeros(time_len1 + 1, 1, self.dx) + self.initial_val

        ## Independent Propagation of two coupled particle paths pairs
        for t in range(self.T):
            start_ind1 = self.getind(t)
            start_ind2 = self.getcind(t)
            update_ind1 = self.getind(t + 1)
            update_ind2 = self.getcind(t + 1)
            # Euler Update
            un1[start_ind1 + 1:update_ind1 + 1], cn1[start_ind2 + 1:update_ind2 + 1] = self.coupled_update(
                un1[start_ind1], cn1[start_ind2])
            un2[start_ind1 + 1:update_ind1 + 1], cn2[start_ind2 + 1:update_ind2 + 1] = self.coupled_update(
                un2[start_ind1], cn2[start_ind2])

        ## Lag-one forward for the first path, note that we input only one pair of coupled particle paths into ccpf kernel
        un1_lag_forward, cn1_lag_forward = self.ccpf_kernel(un1[:, 0, :], cn1[:, 0, :], observe_path)
        four_path = un1_lag_forward, cn1_lag_forward, un2[:, 0, :], cn2[:, 0, :]
        return four_path

    # Two Coupled Finer & Coraser update in unit time, output (2**l, N, dx) (2**(l-1), N, dx) (2**l, N, dx) (2**(l-1), N, dx)
    # Input shape (N, dx) (N, dx) (N, dx) (N, dx)
    def twocoupled_update(self, u1, c1, u2, c2):
        hl = 2 ** (-self.l)
        hlm1 = 2 ** (-self.l + 1)

        # Initialize
        u1_out = torch.zeros(int(2 ** (self.l) + 1), u1.shape[0], u1.shape[-1])
        c1_out = torch.zeros(int(2 ** (self.l - 1) + 1), c1.shape[0], c1.shape[-1])
        u2_out = torch.zeros(int(2 ** (self.l) + 1), u2.shape[0], u2.shape[-1])
        c2_out = torch.zeros(int(2 ** (self.l - 1) + 1), c2.shape[0], c2.shape[-1])

        # Initial values input
        u1_out[0], c1_out[0], u2_out[0], c2_out[0] = u1, c1, u2, c2

        # Coupled Euler Update
        for dt1 in range(2 ** (self.l - 1)):
            dw1 = torch.randn(u1.shape[0], self.dx, 1) * np.sqrt(hl)
            dw2 = torch.randn(u1.shape[0], self.dx, 1) * np.sqrt(hl)
            dw = dw1 + dw2
            u1_out[2 * dt1 + 1] = u1_out[2 * dt1] + self.mu(u1_out[2 * dt1]) * hl + (self.sigma(u1_out[2 * dt1]) @ dw1)[
                ..., 0]
            u1_out[2 * dt1 + 2] = u1_out[2 * dt1 + 1] + self.mu(u1_out[2 * dt1 + 1]) * hl + \
                                  (self.sigma(u1_out[2 * dt1 + 1]) @ dw2)[..., 0]
            c1_out[dt1 + 1] = c1_out[dt1] + self.mu(c1_out[dt1]) * hlm1 + (self.sigma(c1_out[dt1]) @ dw)[..., 0]
            u2_out[2 * dt1 + 1] = u2_out[2 * dt1] + self.mu(u2_out[2 * dt1]) * hl + (self.sigma(u2_out[2 * dt1]) @ dw1)[
                ..., 0]
            u2_out[2 * dt1 + 2] = u2_out[2 * dt1 + 1] + self.mu(u2_out[2 * dt1 + 1]) * hl + \
                                  (self.sigma(u2_out[2 * dt1 + 1]) @ dw2)[..., 0]
            c2_out[dt1 + 1] = c2_out[dt1] + self.mu(c2_out[dt1]) * hlm1 + (self.sigma(c2_out[dt1]) @ dw)[..., 0]

        return u1_out[1:], c1_out[1:], u2_out[1:], c2_out[1:]

    # Get the coupled resampling index through rejection sampling technique of length 'N'
    # Here 'N' does not need to be self.N
    def maximal_rejection_sample_indice(self, weight1, weight2, N):
        ## Step 1
        dice1 = np.random.random_sample(N)
        bins1 = np.cumsum(weight1)
        bins1[-1] = np.max([1, bins1[-1]])
        indice1 = np.digitize(dice1, bins1)
        u_sample = np.random.random_sample(N) * weight1[indice1]

        ## Initialization
        indice2 = np.zeros(indice1.shape).astype(int)

        ## Step 1 Accepted: Identical indices
        step1_accepted = np.where(u_sample <= weight2[indice1])[0]
        indice2[step1_accepted] = indice1[step1_accepted]
        ## Step 1 Rejected
        step1_rejected = np.where(u_sample > weight2[indice1])[0]
        step1_num_rejected = step1_rejected.shape[0]

        ## Step 2
        nrejected = step1_num_rejected
        rejected = step1_rejected

        # step 2 terminate when every indice is accepted
        while (nrejected != 0):
            dice2 = np.random.random_sample(nrejected)
            bins2 = np.cumsum(weight2)
            bins2[-1] = np.max([1, bins2[-1]])
            indice2[rejected] = np.digitize(dice2, bins2)

            ## We only deal with indice2[rejected], which is the particles that got rejected in Step 1.
            v_sample = np.random.random_sample(nrejected) * weight2[indice2[rejected]]

            ## Step 2 Accepted: Sample indice2 independently.
            ## step2_accepted is the index of indice2 that got accepted in Step 2.
            step2_accepted = rejected[np.where(v_sample >= weight1[indice2[rejected]])[0]]

            ## Step 2 Rejected: Repeat Step 2 again
            ## rejected is the index of indice2 that got rejected in Step 2.
            rejected = rejected[np.where(v_sample < weight1[indice2[rejected]])[0]]
            nrejected = rejected.shape[0]

        return indice1, indice2

    # Coupled Maximal Resample 'N' coupled particle pairs, based on (x1,x2) associated with weights (weight1, weight2)
    # Shape of x1: (-1, self.N, self.dx), shape of weight1: (self.N)
    def coupled_maximal_rejection_sample(self, weight1, weight2, x1, x2, N):

        ## Initialization
        x1_hat = torch.zeros(x1.shape)
        x2_hat = torch.zeros(x2.shape)
        ## Sample Indices for both particles collection (N fine particles, N coarse particles)
        indice1, indice2 = self.maximal_rejection_sample_indice(weight1, weight2, N)
        ## Get the resampled
        x1_hat = x1[:, indice1, :]
        x2_hat = x2[:, indice2, :]

        return x1_hat, x2_hat

    # Adaptive Maximally coupled resampling through rejection sampling method
    def coupled_maximal_rejection_resampling(self, weight1, weight2, gn1, gn2, x1, x2):
        ess = 1 / ((weight1 ** 2).sum())
        if ess <= (self.N / 2):
            # When resampling happens, unormalized likelihood function reset
            gn1 = torch.zeros(self.N)
            gn2 = torch.zeros(self.N)
            # Maimal coupled sampling
            x1_hat, x2_hat = self.coupled_maximal_rejection_sample(weight1, weight2, x1, x2, self.N)
        if ess > (self.N / 2):
            x1_hat, x2_hat = x1, x2

        return x1_hat, gn1, x2_hat, gn2

    # indice1 has shape (N), here N does not have to be equal to self.N
    def maximal_fix_index(self, weight1, weight2, indice1, N):

        ## Step 1: Truncated with known indice1
        u_sample = np.random.random_sample(N) * weight1[indice1]

        ## Initialization
        indice2 = np.zeros(indice1.shape).astype(int)

        ## Step 1 Accepted: Identical indices
        step1_accepted = np.where(u_sample <= weight2[indice1])[0]
        indice2[step1_accepted] = indice1[step1_accepted]
        ## Step 1 Rejected
        step1_rejected = np.where(u_sample > weight2[indice1])[0]
        step1_num_rejected = step1_rejected.shape[0]

        ## Step 2
        nrejected = step1_num_rejected
        rejected = step1_rejected

        # step 2 terminate when every indice is accepted
        while (nrejected != 0):
            dice2 = np.random.random_sample(nrejected)
            bins2 = np.cumsum(weight2)
            bins2[-1] = np.max([1, bins2[-1]])
            indice2[rejected] = np.digitize(dice2, bins2)

            ## We only deal with indice2[rejected], which is the particles that got rejected in Step 1.
            v_sample = np.random.random_sample(nrejected) * weight2[indice2[rejected]]

            ## Step 2 Accepted: Sample indice2 independently.
            ## step2_accepted is the index of indice2 that got accepted in Step 2.
            step2_accepted = rejected[np.where(v_sample >= weight1[indice2[rejected]])[0]]

            ## Step 2 Rejected: Repeat Step 2 again
            ## rejected is the index of indice2 that got rejected in Step 2.
            rejected = rejected[np.where(v_sample < weight1[indice2[rejected]])[0]]
            nrejected = rejected.shape[0]

        return indice2

    # Numpy version of coupled maximal resample
    def coupled_maximal_sample_numpy(self, weight1, weight2, x1, x2, N):

        # Initialize
        x1_hat = np.zeros((x1.shape[0], N))
        x2_hat = np.zeros((x2.shape[0], N))

        # Calculating many weights
        unormal_min_weight = np.minimum(weight1, weight2)
        min_weight_sum = np.sum(unormal_min_weight)
        min_weight = unormal_min_weight / min_weight_sum
        unormal_reduce_weight1 = weight1 - unormal_min_weight
        unormal_reduce_weight2 = weight2 - unormal_min_weight

        ## Sample with uniform dice
        dice = np.random.random_sample(N)
        ## [0] takes out the numpy array which is suitable afterwards
        coupled = np.where(dice <= min_weight_sum)[0]
        independ = np.where(dice > min_weight_sum)[0]
        ncoupled = np.sum(dice <= min_weight_sum)
        nindepend = np.sum(dice > min_weight_sum)

        if ncoupled > 0:
            dice1 = np.random.random_sample(ncoupled)
            bins = np.cumsum(min_weight)
            bins[-1] = np.max([1, bins[-1]])
            x1_hat[:, coupled] = x1[:, np.digitize(dice1, bins)]
            x2_hat[:, coupled] = x2[:, np.digitize(dice1, bins)]

        ## nindepend>0 implies min_weight_sum>0 imples np.sum(unormal_reduce_weight*) is positive, thus the division won't report error
        if nindepend > 0:
            reduce_weight1 = unormal_reduce_weight1 / np.sum(unormal_reduce_weight1)
            reduce_weight2 = unormal_reduce_weight2 / np.sum(unormal_reduce_weight2)
            dice2 = np.random.random_sample(nindepend)
            bins1 = np.cumsum(reduce_weight1)
            bins1[-1] = np.max([1, bins1[-1]])
            bins2 = np.cumsum(reduce_weight2)
            bins2[-1] = np.max([1, bins2[-1]])
            x1_hat[:, independ] = x1[:, np.digitize(dice2, bins1)]
            x2_hat[:, independ] = x2[:, np.digitize(dice2, bins2)]

        return x1_hat, x2_hat

    ## We require that indice_len <= weight1.shape[0]
    def coupled_maximal_index_sample(self, weight1, weight2, indice_len):

        N = weight1.shape[0]
        x1 = np.arange(N).reshape((1, N))
        x2 = np.arange(N).reshape((1, N))
        i1, i2 = self.coupled_maximal_sample_numpy(weight1, weight2, x1, x2, indice_len)
        i1 = np.ravel(i1).astype(int)
        i2 = np.ravel(i2).astype(int)
        return i1, i2

    def weight_hat(self, weight1, weight2, indice1, indice2):
        N = weight1.shape[0]
        first_term = np.minimum(weight1[indice1], weight2[indice1])
        second_term = (weight1[indice1] - np.minimum(weight1[indice1], weight2[indice1])) / (
                    1 - np.sum(np.minimum(weight1, weight2)))
        second_term = second_term * (weight2[indice2] - np.minimum(weight1[indice2], weight2[indice2]))
        return first_term + second_term

    # 4-Maximal Resampling Index, returned each indice of length 'N'
    def maximal_of_maximal_four(self, weight1, weight2, weight3, weight4, N):

        ## Step 1
        indice1, indice2 = self.coupled_maximal_index_sample(weight1, weight2, N)
        weight_hat1212 = self.weight_hat(weight1, weight2, indice1, indice2)
        u_sample = np.random.random_sample(N) * weight_hat1212
        weight_hat3412 = self.weight_hat(weight3, weight4, indice1, indice2)

        ## Initialization
        indice3 = np.zeros(indice1.shape).astype(int)
        indice4 = np.zeros(indice2.shape).astype(int)

        ## Step 1 Accepted: Identical indices
        step1_accepted = np.where(u_sample <= weight_hat3412)[0]
        indice3[step1_accepted] = indice1[step1_accepted]
        indice4[step1_accepted] = indice2[step1_accepted]

        ## Step 1 Rejected
        step1_rejected = np.where(u_sample > weight_hat3412)[0]
        step1_num_rejected = step1_rejected.shape[0]

        ## Step 2
        nrejected = step1_num_rejected
        rejected = step1_rejected

        # step 2 terminate when every indice is accepted
        while (nrejected != 0):
            ## Sample indice 3&4 [rejected], which are the particles that got rejected in Step 1.
            indice3[rejected], indice4[rejected] = self.coupled_maximal_index_sample(weight3, weight4, nrejected)
            weight_hat3434 = self.weight_hat(weight3, weight4, indice3[rejected], indice4[rejected])
            v_sample = np.random.random_sample(nrejected) * weight_hat3434
            weight_hat1234 = self.weight_hat(weight1, weight2, indice3[rejected], indice4[rejected])

            ## Step 2 Accepted: Sample indice2 independently.
            ## step2_accepted is the index of indice2 that got accepted in Step 2.
            step2_accepted = rejected[np.where(v_sample >= weight_hat1234)[0]]

            ## Step 2 Rejected: Repeat Step 2 again
            ## rejected is the index of indice2 that got rejected in Step 2.
            rejected = rejected[np.where(v_sample < weight_hat1234)[0]]
            nrejected = rejected.shape[0]

        return indice1, indice2, indice3, indice4

    # 4-Maximal Resample, each output of shape [-1,N,self.dx]
    def four_sample(self, x1, r1, x2, r2, x3, r3, x4, r4, N):

        ## Step1
        if (r1 == r3).all() and (r2 != r4).any():
            indice1, indice2 = self.coupled_maximal_index_sample(r1, r2, N)
            indice3 = indice1
            indice4 = self.maximal_fix_index(r1, r4, indice1, N)
            # Step2
        if (r2 == r4).all() and (r1 != r3).any():
            indice2, indice1 = self.coupled_maximal_index_sample(r2, r1, N)
            indice4 = indice2
            indice3 = self.maximal_fix_index(r2, r3, indice2, N)
        # Step3
        if ((r1 != r3).any() and (r2 != r4).any()) or ((r1 == r3).all() and (r2 == r4).all()):
            indice1, indice2, indice3, indice4 = self.maximal_of_maximal_four(r1, r2, r3, r4, N)

        # Output by picking new particles
        x1_hat = x1[:, indice1, :]
        x2_hat = x2[:, indice2, :]
        x3_hat = x3[:, indice3, :]
        x4_hat = x4[:, indice4, :]

        return x1_hat, x2_hat, x3_hat, x4_hat

        # 4-Maximal Resample with one set of 4 paths output

    def four_sample_one_output(self, x1, r1, x2, r2, x3, r3, x4, r4):
        N = 1
        x1_hat, x2_hat, x3_hat, x4_hat = self.four_sample(x1, r1, x2, r2, x3, r3, x4, r4, N)
        return x1_hat[:, 0, :], x2_hat[:, 0, :], x3_hat[:, 0, :], x4_hat[:, 0, :]

    # Adaptive Resampling of 4 coupled paths each of shape (-1,self.N,self.dx)
    def four_resampling(self, x_fine1, weight_fine1, gn1, x_coarse1, weight_coarse1, gc1, x_fine2, weight_fine2, gn2,
                        x_coarse2, weight_coarse2, gc2):

        ## Adaptive Resampling: Compute the Effective Sample Size (ESS)
        ess_collect = np.zeros(4)
        ess_collect[0] = 1 / ((weight_fine1 ** 2).sum())
        ess_collect[1] = 1 / ((weight_fine2 ** 2).sum())
        ess_collect[2] = 1 / ((weight_coarse1 ** 2).sum())
        ess_collect[3] = 1 / ((weight_coarse2 ** 2).sum())
        ess = np.min(ess_collect)

        if ess > (self.N / 2):
            xf1 = x_fine1
            xf2 = x_fine2
            xc1 = x_coarse1
            xc2 = x_coarse2
        if ess <= (self.N / 2):
            # When resampling happens, unormalized likelihood function reset
            gn1 = torch.zeros(self.N)
            gn2 = torch.zeros(self.N)
            gc1 = torch.zeros(self.N)
            gc2 = torch.zeros(self.N)
            # Maimal coupled sampling
            xf1, xc1, xf2, xc2 = self.four_sample(x_fine1, weight_fine1, x_coarse1, weight_coarse1, x_fine2,
                                                  weight_fine2, x_coarse2, weight_coarse2, self.N)

        return xf1, gn1, xc1, gc1, xf2, gn2, xc2, gc2

    # Coupling of Coupled Conditional Particle Filter Markov Kernel, propagates two coupled particle path (4 particle paths)
    def cccpf_kernel(self, four_path, observe_path):

        fine_path1, coarse_path1, fine_path2, coarse_path2 = four_path
        hl = 2 ** (-self.l)
        time_len = self.getind(self.T)
        time_len1 = self.getcind(self.T)

        ## Initialization
        un1 = torch.zeros(time_len + 1, self.N, self.dx) + self.initial_val
        un1_hat = torch.zeros(time_len + 1, self.N, self.dx) + self.initial_val
        cn1 = torch.zeros(time_len1 + 1, self.N, self.dx) + self.initial_val
        cn1_hat = torch.zeros(time_len1 + 1, self.N, self.dx) + self.initial_val
        un2 = torch.zeros(time_len + 1, self.N, self.dx) + self.initial_val
        un2_hat = torch.zeros(time_len + 1, self.N, self.dx) + self.initial_val
        cn2 = torch.zeros(time_len1 + 1, self.N, self.dx) + self.initial_val
        cn2_hat = torch.zeros(time_len1 + 1, self.N, self.dx) + self.initial_val
        gn1 = torch.zeros(self.N)
        gn2 = torch.zeros(self.N)
        gc1 = torch.zeros(self.N)
        gc2 = torch.zeros(self.N)

        # Transition update and Resampling, Loop over time
        for t in range(self.T):
            start_ind1 = self.getind(t)
            start_ind2 = self.getcind(t)
            update_ind1 = self.getind(t + 1)
            update_ind2 = self.getcind(t + 1)
            un1[:start_ind1 + 1] = un1_hat[:start_ind1 + 1]
            un2[:start_ind1 + 1] = un2_hat[:start_ind1 + 1]
            cn1[:start_ind2 + 1] = cn1_hat[:start_ind2 + 1]
            cn2[:start_ind2 + 1] = cn2_hat[:start_ind2 + 1]

            # Coupled Euler Updata, with same Brownian Motion Path for the two Coupled Paths
            un1[start_ind1 + 1:update_ind1 + 1], cn1[start_ind2 + 1:update_ind2 + 1], un2[
                                                                                      start_ind1 + 1:update_ind1 + 1], cn2[
                                                                                                                       start_ind2 + 1:update_ind2 + 1] = self.twocoupled_update(
                un1[start_ind1], cn1[start_ind2], un2[start_ind1], cn2[start_ind2])

            ## The last particle is fixed, and it joins the resampling process
            un1[:, -1, :] = fine_path1
            cn1[:, -1, :] = coarse_path1
            un2[:, -1, :] = fine_path2
            cn2[:, -1, :] = coarse_path2

            ## Accumulating Weight Function
            gn1 = self.llg(un1[update_ind1], observe_path[t + 1]) + gn1
            what1 = torch.exp(gn1 - torch.max(gn1))
            wn1 = what1 / torch.sum(what1)
            wn1 = wn1.detach().numpy()

            gc1 = self.llg(cn1[update_ind2], observe_path[t + 1]) + gc1
            wchat1 = torch.exp(gc1 - torch.max(gc1))
            wc1 = wchat1 / torch.sum(wchat1)
            wc1 = wc1.detach().numpy()

            gn2 = self.llg(un2[update_ind1], observe_path[t + 1]) + gn2
            what2 = torch.exp(gn2 - torch.max(gn2))
            wn2 = what2 / torch.sum(what2)
            wn2 = wn2.detach().numpy()

            gc2 = self.llg(cn2[update_ind2], observe_path[t + 1]) + gc2
            wchat2 = torch.exp(gc2 - torch.max(gc2))
            wc2 = wchat2 / torch.sum(wchat2)
            wc2 = wc2.detach().numpy()

            ## Four Indices Resampling
            un1_hat[:update_ind1 + 1], gn1, cn1_hat[:update_ind2 + 1], gc1, un2_hat[:update_ind1 + 1], gn2, cn2_hat[
                                                                                                            :update_ind2 + 1], gc2 = self.four_resampling(
                un1[:update_ind1 + 1], wn1, gn1, cn1[:update_ind2 + 1], wc1, gc1, un2[:update_ind1 + 1], wn2, gn2,
                cn2[:update_ind2 + 1], wc2, gc2)
            un1_hat[:, -1, :] = fine_path1
            cn1_hat[:, -1, :] = coarse_path1
            un2_hat[:, -1, :] = fine_path2
            cn2_hat[:, -1, :] = coarse_path2

        ## Four Sampling
        four_path_output = self.four_sample_one_output(un1, wn1, cn1, wc1, un2, wn2, cn2, wc2)

        return four_path_output

    # Generate a chain of two coupled particle pahts, with CCCPF of length 'num_step+1'
    # Intitial posision of one coupled particle path is laged-one forward
    def chain_gen_cccpf(self, num_step, observe_path):
        u1_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx)
        c1_chain = torch.zeros(num_step + 1, self.getcind(self.T) + 1, self.dx)
        u2_chain = torch.zeros(num_step + 1, self.getind(self.T) + 1, self.dx)
        c2_chain = torch.zeros(num_step + 1, self.getcind(self.T) + 1, self.dx)
        u1_chain[0], c1_chain[0], u2_chain[0], c2_chain[0] = self.initial_lag_4path(observe_path)

        for step in range(num_step):
            four_input = u1_chain[step], c1_chain[step], u2_chain[step], c2_chain[step]
            u1_chain[step + 1], c1_chain[step + 1], u2_chain[step + 1], c2_chain[step + 1] = self.cccpf_kernel(
                four_input, observe_path)
        return u1_chain, c1_chain, u2_chain, c2_chain


"b_batched obtained with shape (Batch_size, dim_x, 1)"


def b_torch(podpdo, xt):
    batched_sigma = podpdo.sigma(xt)
    batched_inv_sigma = torch.inverse(batched_sigma)
    batched_a = podpdo.mu(xt)
    batched_a = torch.unsqueeze(batched_a, -1)
    return torch.matmul(batched_inv_sigma, batched_a)


"Discrete F function"


def Ffunc(podpdo, x_discrete, observe_path):
    "We will put these two into params of the class object later"
    T = observe_path.shape[0] - 1
    l = int(np.log2((x_discrete.shape[0] - 1) / T))

    dt = 2 ** (-l)
    xt = x_discrete[:-1]
    x_seq = x_discrete[0::2 ** l][1:]
    y_seq = observe_path[1:]

    dxtmat = torch.unsqueeze(x_discrete[1:] - x_discrete[:-1], -1)
    b_batched = b_torch(podpdo, xt)
    bt_batched = torch.transpose(b_batched, -1, -2)
    s_batched = podpdo.sigma(xt)
    sinv_batched = torch.inverse(s_batched)

    val1 = torch.sum(torch.matmul(bt_batched, b_batched) * dt, axis=0)
    val1 = torch.squeeze(val1)
    val2 = torch.sum(torch.matmul(bt_batched, torch.matmul(sinv_batched, dxtmat)), axis=0)
    val2 = torch.squeeze(val2)
    val3 = torch.sum(podpdo.likelihood_logscale(x_seq, y_seq))
    return -0.5 * val1 + val2 + val3


"Discretized G function"


def Gl_auto(podpdo, x_discrete, observe_path):
    theta = podpdo.theta

    def F_reduce(theta):
        podpdo.theta = theta
        return Ffunc(podpdo, x_discrete, observe_path)

    return jac(F_reduce, theta)


"Discretized G @ G.T value"


def GlGl_auto(podpdo, x_discrete, observe_path):
    gl = torch.unsqueeze(Gl_auto(podpdo, x_discrete, observe_path), axis=-1)
    return gl @ gl.T


"Discretized K function"


def Kl_auto(podpdo, x_discrete, observe_path):
    theta = podpdo.theta

    def F_reduce(theta):
        podpdo.theta = theta
        return Ffunc(podpdo, x_discrete, observe_path)

    return hessian(F_reduce, theta)


"""
Using lag-one & Driving CCPF to debias wrt Conditional Particle Filter's estimate
of certain function's expectation wrt to smoothing distribution. Such function takes
input : podpdo model class, discrete signal path, observation path
"""


def func_debias_cpf(condpf, observe_path, func, cut_spot=1):
    # path1 is the lag-one forward path
    path1_col = []
    path2_col = []
    path1, path2 = condpf.initial_lag_2path(observe_path)
    path1_col.append(path1)
    path2_col.append(path2)
    ## Loop untill meeting time
    meeting_time = 0
    while (path1 != path2).any():
        meeting_time += 1
        path1, path2 = condpf.drive_ccpf_kernel(path1, path2, observe_path)
        path1_col.append(path1)
        path2_col.append(path2)

    if cut_spot > meeting_time:
        print('Error: Cut_spot too small for meeting time {}'.format(meeting_time))
        func_output = 0

    if cut_spot <= meeting_time:
        func_output = 0
        func_output += func(condpf.model, path1_col[cut_spot], observe_path)
        if cut_spot < meeting_time:
            ## If cut_spot = meeting_time -1, this for loop won't occur at all, essentially used when cut_spot <= meeting_time -2.
            for m in range(cut_spot + 1, meeting_time):
                func_output += (func(condpf.model, path1_col[m], observe_path) - func(condpf.model, path2_col[m],
                                                                                      observe_path))
    return func_output


"""
Single CondPF chain lag-one debias with two functional output
"""


def twofunc_debias_cpf(condpf, observe_path, func1, func2, cut_spot=1):
    # path1 is the lag-one forward path
    path1_col = []
    path2_col = []
    path1, path2 = condpf.initial_lag_2path(observe_path)
    path1_col.append(path1)
    path2_col.append(path2)
    ## Loop untill meeting time
    meeting_time = 0
    while (path1 != path2).any():
        meeting_time += 1
        path1, path2 = condpf.drive_ccpf_kernel(path1, path2, observe_path)
        path1_col.append(path1)
        path2_col.append(path2)

    if cut_spot > meeting_time:
        print('Error: Cut_spot too small for meeting time {}'.format(meeting_time))
        func_output = 0

    if cut_spot <= meeting_time:
        func_output1 = 0
        func_output2 = 0
        func_output1 += func1(condpf.model, path1_col[cut_spot], observe_path)
        func_output2 += func2(condpf.model, path1_col[cut_spot], observe_path)
        if cut_spot < meeting_time:
            ## If cut_spot = meeting_time -1, this for loop won't occur at all, essentially used when cut_spot <= meeting_time -2.
            for m in range(cut_spot + 1, meeting_time):
                func_output1 += (func1(condpf.model, path1_col[m], observe_path) - func1(condpf.model, path2_col[m],
                                                                                         observe_path))
                func_output2 += (func2(condpf.model, path1_col[m], observe_path) - func2(condpf.model, path2_col[m],
                                                                                         observe_path))
    return func_output1, func_output2


"""
Debias of coupled difference of certain function wrt Coupled Conditional Particle Filter
through C-CCPF and lag-one structure. We eliminate the MCMC bias but not the discretization 
bias here
"""


def diff_func_debias_ccpf(condpf, observe_path, func, cut_spot=1):
    ## path1 & path2 is the lag-one forward paths pair
    path1_col = []
    path2_col = []
    path3_col = []
    path4_col = []
    path1, path2, path3, path4 = condpf.initial_lag_4path(observe_path)
    path1_col.append(path1)
    path2_col.append(path2)
    path3_col.append(path3)
    path4_col.append(path4)
    ## Loop untill meeting_time1 for two fine paths: path1 & path3
    mtime13 = 0
    mtime24 = 0
    while (path1 != path3).any() or (path2 != path4).any():
        ## Increase repective meeting time if coupled paths are different
        mtime13 += int((path1 != path3).any())
        mtime24 += int((path2 != path4).any())
        ## Markov Transition
        fourpath_input = path1, path2, path3, path4
        path1, path2, path3, path4 = condpf.cccpf_kernel(fourpath_input, observe_path)
        ## Store the values
        path1_col.append(path1)
        path2_col.append(path2)
        path3_col.append(path3)
        path4_col.append(path4)

    ## Func^l computation
    if cut_spot > mtime13:
        print('Error: Cut_spot too small for fine meet_time {}'.format(mtime13))
    if cut_spot <= mtime13:
        value_fine = 0
        value_fine += func(condpf.model, path1_col[cut_spot], observe_path)
        if cut_spot < mtime13:
            for m in range(cut_spot + 1, mtime13):
                value_fine += (func(condpf.model, path1_col[m], observe_path) - func(condpf.model, path3_col[m],
                                                                                     observe_path))

    ## Func^{l-1} computation
    if cut_spot > mtime24:
        print('Error: Cut_spot too small for coarse meet_time {}'.format(mtime24))
    if cut_spot <= mtime24:
        value_coarse = 0
        value_coarse += func(condpf.model, path2_col[cut_spot], observe_path)
        if cut_spot < mtime24:
            for m in range(cut_spot + 1, mtime24):
                value_coarse += (func(condpf.model, path2_col[m], observe_path) - func(condpf.model, path4_col[m],
                                                                                       observe_path))

    coupled_difference = value_fine - value_coarse
    return coupled_difference


"Estimate of the Score"


def score_est(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe = torch.zeros((rep_num, dim_theta))
    for i in range(rep_num):
        repe[i] = func_debias_cpf(condpf, observe_path, Gl_auto)
    return torch.mean(repe, axis=0)


"Estimate of Score with progbar"


def score_est_with_progbar(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe = torch.zeros((rep_num, dim_theta))
    pr = progressbar.ProgressBar(max_value=rep_num).start()
    for i in range(rep_num):
        repe[i] = func_debias_cpf(condpf, observe_path, Gl_auto)
        pr.update(i + 1)
    pr.finish()
    print('Level', condpf.l, 'score estimation finised')
    return torch.mean(repe, axis=0)


"Slow & Bad 3-Term Estimate of the Hessian"


def hessian_slow_est(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand3 = torch.zeros((rep_num, dim_theta, dim_theta))
    for i in range(rep_num):
        repe_summand1[i] = func_debias_cpf(condpf, observe_path, Kl_auto)
        repe_summand2[i] = func_debias_cpf(condpf, observe_path, GlGl_auto)
        inter_val = torch.unsqueeze(func_debias_cpf(condpf, observe_path, Gl_auto), axis=-1)
        repe_summand3[i] = inter_val @ inter_val.T
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.mean(repe_summand2, axis=0)
    val3 = torch.mean(repe_summand3, axis=0)
    return val1 + val2 - val3


"Slow & Bad 3-Term Estimate of the Hessian with Porgbar"


def hessian_slow_est_with_progbar(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand3 = torch.zeros((rep_num, dim_theta, dim_theta))
    pr = progressbar.ProgressBar(max_value=rep_num).start()
    for i in range(rep_num):
        repe_summand1[i] = func_debias_cpf(condpf, observe_path, Kl_auto)
        repe_summand2[i] = func_debias_cpf(condpf, observe_path, GlGl_auto)
        inter_val = torch.unsqueeze(func_debias_cpf(condpf, observe_path, Gl_auto), axis=-1)
        repe_summand3[i] = inter_val @ inter_val.T
        pr.update(i + 1)
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.mean(repe_summand2, axis=0)
    val3 = torch.mean(repe_summand3, axis=0)
    print('Level', l, '3-term Hessian estimation finised')
    return val1 + val2 - val3


"""
In order to Speed-up Hessian Estimation, we use same ConPF chain for its estimation, this
can literally bring down the cost of getting Hessin to same level of getting Jacobian 
Estimation, which can actually be a great advantage of our algorithm: Estimation of higher-order
derivative can be obtained with essentially same cost order. As opposed to the square cost for most
methods out there. (The most cost here has nothing to do with 'differentiation' or 'evaluation of 
d*d elements instead of d elements', but the MCMC chain generation, whose cost is absolutely the 
same for Hessian & Jacobian estimation scheme)

From the OU model, we verifies that this approach is both cheap and provides very accurate result!
"""

"Estimate of the Hessian with cheaper cost"


def hessian_data_logprob_est(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta))
    for i in range(rep_num):
        repe_summand1[i], repe_summand2[i] = twofunc_debias_cpf(condpf, observe_path, Kl_auto, Gl_auto)
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.tensor(np.cov(repe_summand2.T)).float()
    return val1 + val2


"Estimate of the Hessian with cheaper cost"


def hessian_data_logprob_est_progbar(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta))
    pbar = pkbar.Pbar(name='Hessian estimate calculating', target=rep_num)
    for i in range(rep_num):
        repe_summand1[i], repe_summand2[i] = twofunc_debias_cpf(condpf, observe_path, Kl_auto, Gl_auto)
        pbar.update(i)
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.tensor(np.cov(repe_summand2.T)).float()
    return val1 + val2


## Function to evaluate L2 distance between the direction (normalized vector) of actual and estimated gradient,
## as only the direction is used in gradient descend algorithm
def acc_grad(a, b):
    norm_a = a / torch.sqrt(np.sum(a ** 2))
    norm_b = b / torch.sqrt(np.sum(b ** 2))
    return 1 - torch.sqrt(torch.sum((norm_a - norm_b) ** 2))


## Use relative L2 distance between true paramters and the learned paramters to evaluate acc_theta
def acc_param(true, estimate):
    upper = torch.sqrt(torch.sum((true - estimate) ** 2))
    lower = torch.sqrt(torch.sum(true ** 2))
    return 1 - upper / lower


"""
We need to include the Coupled Difference R&G estimation structure into the pacakage
"""
"""
Debias of coupled difference of two function sharing same paths wrt Coupled Conditional
Particle Filter through C-CCPF and lag-one structure. We eliminate the MCMC bias but not
the discretization bias here
"""


def twodiff_func_debias_ccpf(condpf, observe_path, func, func1, cut_spot=1):
    ## path1 & path2 is the lag-one forward paths pair
    path1_col = []
    path2_col = []
    path3_col = []
    path4_col = []
    path1, path2, path3, path4 = condpf.initial_lag_4path(observe_path)
    path1_col.append(path1)
    path2_col.append(path2)
    path3_col.append(path3)
    path4_col.append(path4)
    ## Loop untill meeting_time1 for two fine paths: path1 & path3
    mtime13 = 0
    mtime24 = 0
    while (path1 != path3).any() or (path2 != path4).any():
        ## Increase repective meeting time if coupled paths are different
        mtime13 += int((path1 != path3).any())
        mtime24 += int((path2 != path4).any())
        ## Markov Transition
        fourpath_input = path1, path2, path3, path4
        path1, path2, path3, path4 = condpf.cccpf_kernel(fourpath_input, observe_path)
        ## Store the values
        path1_col.append(path1)
        path2_col.append(path2)
        path3_col.append(path3)
        path4_col.append(path4)

    ## Func^l computation
    if cut_spot > mtime13:
        print('Error: Cut_spot too small for fine meet_time {}'.format(mtime13))
    if cut_spot <= mtime13:
        value_fine = 0
        value_fine1 = 0
        value_fine += func(condpf.model, path1_col[cut_spot], observe_path)
        value_fine1 += func1(condpf.model, path1_col[cut_spot], observe_path)
        if cut_spot < mtime13:
            for m in range(cut_spot + 1, mtime13):
                value_fine += (func(condpf.model, path1_col[m], observe_path) - func(condpf.model, path3_col[m],
                                                                                     observe_path))
                value_fine1 += (func1(condpf.model, path1_col[m], observe_path) - func1(condpf.model, path3_col[m],
                                                                                        observe_path))

    ## Func^{l-1} computation
    if cut_spot > mtime24:
        print('Error: Cut_spot too small for coarse meet_time {}'.format(mtime24))
    if cut_spot <= mtime24:
        value_coarse = 0
        value_coarse1 = 0
        value_coarse += func(condpf.model, path2_col[cut_spot], observe_path)
        value_coarse1 += func1(condpf.model, path2_col[cut_spot], observe_path)
        if cut_spot < mtime24:
            for m in range(cut_spot + 1, mtime24):
                value_coarse += (func(condpf.model, path2_col[m], observe_path) - func(condpf.model, path4_col[m],
                                                                                       observe_path))
                value_coarse1 += (func1(condpf.model, path2_col[m], observe_path) - func1(condpf.model, path4_col[m],
                                                                                          observe_path))

    coupled_difference = value_fine - value_coarse
    coupled_difference1 = value_fine1 - value_coarse1
    return coupled_difference, coupled_difference1


"""
Estimate of the Coupled Difference of Jacobian over consecutive discretization level
"""


def score_coupled_diff(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe = torch.zeros((rep_num, dim_theta))
    pr = progressbar.ProgressBar(max_value=rep_num).start()
    for i in range(rep_num):
        repe[i] = diff_func_debias_ccpf(condpf, observe_path, Gl_auto)
        pr.update(i + 1)
    pr.finish()
    print('Level', condpf.l, 'score estimation finised')
    return torch.mean(repe, axis=0)


"""
Estimate of the Coupled Difference of Hessian over Consecutive Discretization Level
Question is the Covariance structure is not guarenteed to keep the couppled difference
variance decaying wrt to increasing discretization level 

Not sure how it works, I am guessing it would be pretty good since coupling exists in there
for sure. Maybe theoretical justification of the 3-term coupling is easier.
"""


def hessian_coupled_diff_progbar(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta))
    pbar = pkbar.Pbar(name='Hessian estimate calculating', target=rep_num)
    for i in range(rep_num):
        repe_summand1[i], repe_summand2[i] = twodiff_func_debias_ccpf(condpf, observe_path, Kl_auto, Gl_auto)
        pbar.update(i)
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.tensor(np.cov(repe_summand2.T)).float()
    return val1 + val2


def hessian_coupled_diff(condpf, observe_path, rep_num):
    dim_theta = condpf.model.theta.shape[0]
    repe_summand1 = torch.zeros((rep_num, dim_theta, dim_theta))
    repe_summand2 = torch.zeros((rep_num, dim_theta))
    for i in range(rep_num):
        repe_summand1[i], repe_summand2[i] = twodiff_func_debias_ccpf(condpf, observe_path, Kl_auto, Gl_auto)
    val1 = torch.mean(repe_summand1, axis=0)
    val2 = torch.tensor(np.cov(repe_summand2.T)).float()
    return val1 + val2


"PMF for l, used in the Rhee & Glynn Estimator, truncated at max_l level"


def pmf_l(max_l):
    pl = 1 / 2 ** (torch.arange(max_l + 1))
    pl = pl / torch.sum(pl)
    return pl


"N_{l} for Hessian increment, since it has static value, we treat it as a scalar"


def nl_increment(max_l):
    return max_l * 2 ** (2 * max_l)


"Rhee & Glynn Estimate of Hessian, truncated at max_l level, here we used condpf.l as default"
"Tuning is required for M to banlance variance with bias"


def RGhessian(condpf, observe_path, M):
    max_l = condpf.l
    Nl = 10
    pl = pmf_l(max_l)
    dim_theta = condpf.model.theta.shape[0]
    rep_estimate = torch.zeros((M, dim_theta, dim_theta))

    pbar = pkbar.Pbar(name='R&G estimate of Hessian calculating', target=M, width=12)
    for m in range(M):
        dist_l = Categorical(pl)
        l = int(dist_l.sample())
        condpf.l = l
        if l == 0:
            rep_estimate[m] = hessian_data_logprob_est(condpf, observe_path, Nl) / pl[0]
        if l != 0:
            rep_estimate[m] = hessian_coupled_diff(condpf, observe_path, Nl) / pl[l]
        pbar.update(m)
    return torch.mean(rep_estimate, axis=0)