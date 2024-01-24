from utils import init_gaussian, init_uniform

from operator import itemgetter

import torch

# used in ngc_classify
class GNCN_PDH_Classify:
    def __init__(self, config, device=None):
        # use itemgetter
        L, dims, fns_phi, fns_g, weight_stddev, beta, leak, use_skip, use_lateral, use_err_precision = itemgetter(
            'L', 'dims', 'fns_phi', 'fns_g', 'weight_stddev', 'beta', 'leak', 'use_skip', 'use_lateral', 'use_err_precision')(config)

        self.L = L
        self.dims = dims
        self.fns_phi = list(map(self.get_activation_fn, fns_phi))
        self.fns_g = list(map(self.get_activation_fn, fns_g))
        self.beta = beta
        self.leak = leak
        self.use_skip = use_skip
        self.use_lateral = use_lateral
        self.use_err_precision = use_err_precision

        self.device = torch.device('cpu') if device is None else device

        self.W = [init_gaussian([dims[i+1], dims[i]], weight_stddev, self.device) for i in range(L)]
        self.E = [init_gaussian([dims[i], dims[i+1]], weight_stddev, self.device) for i in range(L)]

        self.M = []
        if self.use_skip:
            self.M = [init_gaussian([dims[i+2], dims[i]], weight_stddev, self.device) for i in range(L-1)]

        # in ngc-learn implementation, e^0 doesnt use a precision matrix. we therefore create
        # precision matrices for only e^1 to e^{L-1}
        self.Sigma = []
        self.Prec = []
        init_sigma_var = 0.01
        if self.use_err_precision:
            self.Sigma = [init_uniform([dims[i], dims[i]], -init_sigma_var, init_sigma_var, self.device) for i in range(1, L)]
            self.Prec = [None for _ in range(L-1)]
            self.compute_precisions()

        self.V = []
        self.wta_K_top = 18
        self.wta_K_hid = 12
        self.wta_inh = 0.1
        self.wta_exc = 0.15
        # if self.use_lateral:
        #     self.V = ([make_lkwta(dim_hid, self.wta_K_hid, self.wta_inh, self.wta_exc).to(self.device)]
        #         + [make_lkwta(dim_hid, self.wta_K_hid, self.wta_inh, self.wta_exc).to(self.device) for _ in range(L-2)]
        #         + [make_lkwta(dim_top, self.wta_K_top, self.wta_inh, self.wta_exc).to(self.device)])


        self.clip_weights()

    def get_activation_fn(self, name):
        if name == 'relu':
            return torch.relu
        elif name == 'sigmoid':
            return torch.sigmoid
        elif name == 'identity':
            return lambda x: x
        elif name == 'tanh':
            return torch.tanh
        elif name == 'softmax':
            return lambda x: torch.softmax(x, dim=1)
        else:
            raise NotImplementedError(f"Activation function {name} not supported.")


    def parameters(self):
        return self.W + self.E + self.M + self.Sigma

    def state_dict(self):
        state = {}
        for l in range(self.L):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        if self.use_skip:
            for l in range(self.L - 1):
                state[f'M{l}'] = self.M[l]
        return state

    def load_state_dict(self, state):
        for l in range(self.L):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']
        if self.use_skip:
            for l in range(self.L - 1):
                state[f'M{l}'] = self.M[l]


    def infer(self, x_bot, x_top, K=50):
        batch_size = x_top.shape[0]
        z = [x_bot]
        e = [torch.zeros([batch_size, self.dims[0]], device=self.device)]
        for i in range(1, self.L):
            z.append(torch.zeros([batch_size, self.dims[i]], device=self.device))
            e.append(torch.zeros([batch_size, self.dims[i]], device=self.device))
        z.append(x_top)
        # e[L] is a dummy tensor that is initialized to zero and never updated
        # e.append(torch.zeros([batch_size, self.dims[-1]], device=self.device))

        mu = [None for _ in range(self.L)]
        e_out = [e[i] for i in range(len(e))]
        z_out = [self.fns_phi[i](z[i]) for i in range(len(z))]

        for _ in range(K):
            for i in range(1, self.L):
                di = e_out[i-1] @ self.E[i-1] - e_out[i]
                vi = 0.
                # if self.use_lateral:
                #     vi = z_out[i] @ self.V[i-1]
                z[i] += self.beta * (-self.leak * z[i] + di - vi)
                z_out[i] = self.fns_phi[i](z[i])

            for i in range(0, self.L):
                mu_W_input = z_out[i+1] @ self.W[i]
                if self.use_skip and i < self.L - 1:
                    mu[i] = self.fns_g[i](mu_W_input + z_out[i+2] @ self.M[i])
                else:
                    mu[i] = self.fns_g[i](mu_W_input)
                e[i] = z_out[i] - mu[i]
                if self.use_err_precision and i > 0:
                    e_out[i] = e[i] @ self.Prec[i-1]
                else:
                    e_out[i] = e[i]

        self.z = z
        self.z_out = z_out
        self.e = e
        self.e_out = e_out

        return mu[0]

    def calc_updates(self):
        batch_size = self.z[0].shape[0]
        avg_factor = -1.0 / (batch_size)

        for l in range(0, self.L):
            dWl = self.z_out[l+1].T @ self.e_out[l]
            dWl = avg_factor * dWl
            dEl = dWl.T
            self.W[l].grad = dWl
            self.E[l].grad = dEl

        if self.use_skip:
            for l in range(0, self.L - 1):
                dMl = self.z_out[l+2].T @ self.e_out[l]
                dMl = avg_factor * dMl
                self.M[l].grad = dMl

        if self.use_err_precision:
            for l in range(0, self.L - 1):
                Bl = (self.e[l+1].T @ self.e[l+1])
                dSigmal = (Bl - self.Prec[l]) * 0.5
                dSigmal = avg_factor * dSigmal
                self.Sigma[l].grad = dSigmal


    def compute_precisions(self, eps = 0.00025):
        if self.use_err_precision:
            for l in range(self.L - 1):
                Il = torch.eye(self.Sigma[l].shape[1], device=self.device)

                # ensure diagonals are at least 1
                sigmal = self.Sigma[l]
                varl = torch.maximum(torch.tensor(1.0), sigmal) * Il
                sigmal = varl + (sigmal * (1.0 - Il)) + eps
                self.Sigma[l].copy_(sigmal)
                Ll = torch.linalg.cholesky(self.Sigma[l])
                self.Prec[l] = torch.linalg.solve_triangular(Ll, Il, upper=False)



    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))

        if self.use_skip:
            for l in range(self.L - 1):
                Ml_col_norms = self.M[l].norm(dim=0, keepdim=True)
                self.M[l].copy_(self.M[l] / torch.maximum(Ml_col_norms, torch.tensor(1.0)))

        if self.use_err_precision:
            for l in range(self.L - 1):
                Sigmal_col_norms = self.Sigma[l].norm(dim=0, keepdim=True)
                self.Sigma[l].copy_(self.Sigma[l] / torch.maximum(Sigmal_col_norms, torch.tensor(1.0)))


    def project(self, x_top):
        zbar_prev = x_top
        mu_W_input = self.fns_phi[-1](zbar_prev) @ self.W[self.L - 1]
        zbar = self.fns_g[-1](mu_W_input)

        for l in range(self.L - 2, 0, -1):
            mu_W_input = self.fns_phi[l](zbar) @ self.W[l]
            mu_M_input = 0.
            if self.use_skip:
                mu_M_input = self.fns_phi[l+1](zbar_prev) @ self.M[l]
            zbar_prev = zbar
            zbar = self.fns_g[l](mu_W_input + mu_M_input)

        mu_W_input = self.fns_phi[0](zbar) @ self.W[0]
        mu_M_input = 0.
        if self.use_skip:
            mu_M_input = self.fns_phi[1](zbar_prev) @ self.M[0]
        zbar = self.fns_g[0](mu_W_input + mu_M_input)
        return zbar



# implements NGC pseudocode from ANGC appendix
class NGC_ANGC:
    def __init__(self, L, dims, weight_stddev, beta=0.1, beta_e=0.5, gamma=0.001, err_update_coeff=0.95, fn_phi_name='relu', device=None):
        assert len(dims) == L + 1
        self.L = L
        self.dims = dims
        self.beta = beta
        self.beta_e = beta_e
        self.gamma = gamma # leak coefficient
        self.err_update_coeff = err_update_coeff

        self.device = torch.device('cpu') if device is None else device

        # assume dims is in order (bottom, ... hiddens ..., top)
        self.W = []
        for i in range(L):
            self.W.append(init_gaussian([dims[i+1], dims[i]], weight_stddev, self.device))

        # the paper shows E^L, but this would only be required if we updated z^L, which I think we don't
        # (x is clamped to it, it appears to be an error in the paper)
        self.E = []
        for i in range(L-1):
            self.E.append(init_gaussian([dims[i], dims[i+1]], weight_stddev, self.device))

        if fn_phi_name == 'relu':
            self.fn_phi = torch.relu
        else:
            raise NotImplementedError("Only relu is supported for phi.")

        self.fn_g = lambda x: x


    def parameters(self):
        return self.W + self.E

    def state_dict(self):
        state = {}
        for l in range(self.L - 1):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        state[f'W{self.L - 1}'] = self.W[self.L - 1]
        return state

    def load_state_dict(self, state):
        for l in range(self.L - 1):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']
        self.W[self.L - 1] = state[f'W{self.L - 1}']


    def project(self, x_top):
        zbar = x_top
        for i in range(self.L - 1, -1, -1):
            zbar = self.fn_g(self.fn_phi(zbar) @ self.W[i])
        return torch.softmax(zbar, dim=1)

    def infer(self, x_bot, x_top, K=50):
        batch_size = x_bot.shape[0]
        z = [x_bot]
        e = [x_bot - 0.]
        for l in range(1, self.L):
            z.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
            e.append(torch.zeros([batch_size, self.dims[l]], device=self.device))
        z.append(x_top)

        z_out = [None for _ in range(self.L+1)]
        mu = [None for _ in range(self.L)]

        for _ in range(K):
            # d^1 = -e^1 + E^1 e^0
            # z^1 = z^1 + beta * (-gamma_v z^1 + d^1)
            # ...
            # d^{L-1} = -e^{L-1} + E^L e^{L-1}
            # z^{L-1} = z^{L-1} + beta * (-gamma_v z^{L-1} + d^{L-1})
            for i in range(1, self.L):
                di = -e[i] + e[i-1] @ self.E[i-1]
                z[i] += self.beta * (-self.gamma * z[i] + di)
                z_out[i] = self.fn_phi(z[i])
            z_out[self.L] = self.fn_phi(z[self.L])

            mu[0] = self.fn_g(z_out[1] @ self.W[0])
            e[0] = (z[0] - mu[0]) / (2.0 * self.beta_e)
            for i in range(1, self.L):
                mu[i] = self.fn_g(z_out[i+1] @ self.W[i])
                e[i] = (z_out[i] - mu[i]) / (2.0 * self.beta_e)


        self.z = z
        self.z_out = z_out
        self.e = e

        return mu[0]

    def calc_updates(self):
        batch_size = self.z[0].shape[0]
        avg_factor = -1.0 / (batch_size)

        for l in range(0, self.L):
            dWl = self.z_out[l+1].T @ self.e[l]
            dWl = avg_factor * dWl
            self.W[l].grad = dWl
            if l < self.L - 1:
                dEl = self.err_update_coeff * dWl.T
                self.E[l].grad = dEl


    # the weight clipping function from ngc-learn
    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L - 1):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))

        Wl_col_norms = self.W[self.L-1].norm(dim=0, keepdim=True)
        self.W[self.L-1].copy_(self.W[self.L-1] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))

    # implements W^l = 2 W^l / ||W^l|| + c_eps
    def normalize_weights(self, c_eps=1e-6):
        for l in range(self.L - 1):
            self.W[l].copy_(2.0 * self.W[l] / (self.W[l].norm() + c_eps))
            self.E[l].copy_(2.0 * self.E[l] / (self.E[l].norm() + c_eps))
        self.W[self.L - 1].copy_(2.0 * self.W[self.L - 1] / (self.W[self.L - 1].norm() + c_eps))


    def calc_total_discrepancy(self):
        return sum([torch.sum(e**2) for e in self.e[:self.L]])