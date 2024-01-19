from utils import init_gaussian, init_uniform, make_moving_collate_fn, make_lkwta, set_seed

from operator import itemgetter
import os
import pickle
import urllib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision

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




def preprocess_mnist(batch_size, device):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)

    moving_collate = make_moving_collate_fn(device)

    # split into train and validation
    data_train, data_val = random_split(data_train, [50000, 10000])

    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=moving_collate)
    return loader_train, loader_val


def download_mnist1d(data_path):
    url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
    with urllib.request.urlopen(url) as response, open(data_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def preprocess_mnist1d(batch_size, device):
    data_path = './data/mnist1d_data.pkl'
    if not os.path.exists(data_path):
        data_dir = os.path.dirname(data_path)
        os.makedirs(data_dir, exist_ok=True)
        download_mnist1d(data_path)

    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    Y_train = torch.tensor(data['y'], dtype=torch.int64)
    # X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    # Y_test = torch.tensor(data['y_test'], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)

    moving_collate = make_moving_collate_fn(device)

    valid_frac = 0.1
    valid_size = int(len(train_dataset) * valid_frac)
    train_size = len(train_dataset) - valid_size
    data_train, data_val = random_split(train_dataset, [train_size, valid_size])

    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=moving_collate)

    return loader_train, loader_val


def cross_entropy_loss(targets, predictions, eps=1e-7):
    clamped_predictions = torch.clamp(predictions, min=eps, max=1.0 - eps)
    return -torch.sum(targets * torch.log(clamped_predictions))

def eval_model(model, loader, num_classes):
    num_samples = 0
    tot_ce_loss = 0.
    tot_correct = 0
    for (inputs, targets) in loader:
        inputs = inputs.view([-1, model.dims[-1]])
        preds = model.project(inputs)
        targets_oh = F.one_hot(targets, num_classes=num_classes)
        tot_ce_loss += cross_entropy_loss(targets_oh, preds)
        tot_correct += torch.sum(torch.argmax(preds, dim=1) == targets)
        num_samples += inputs.shape[0]
    avg_ce_loss = tot_ce_loss / (1.0 * num_samples)
    acc = tot_correct / (1.0 * num_samples)
    print(f"(Eval)  Avg CE loss = {avg_ce_loss}, Avg accuracy: {acc}")
    return avg_ce_loss, acc



def run_ngc(seed, trial_name='ngc'):
    set_seed(seed)

    num_epochs = 50
    num_classes = 10
    batch_size = 256
    lr = 0.001
    dim_inp = 784
    dim_hid = 25
    K = 60
    # err_update_coeff = 0.95

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # loader_train, loader_val = preprocess_mnist(batch_size, device)
    loader_train, loader_val = preprocess_mnist1d(batch_size, device)

    ngc_config = {
        'L': 3,
        'dims': [num_classes, dim_hid, dim_hid, dim_inp],
        'fns_phi': ['identity', 'relu', 'relu', 'identity'],
        'fns_g': ['softmax', 'identity', 'identity', 'identity'],
        'weight_stddev': 0.025,
        'beta': 0.1,
        'leak': 0.001,
        'use_skip': True,
        'use_lateral': False,
        'use_err_precision': True,
    }

    model = GNCN_PDH_Classify(ngc_config, device=device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, maximize=False)

    val_ce_loss, val_acc = eval_model(model, loader_val, num_classes)
    best_val_ce_loss = val_ce_loss

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch}")
        num_samples = 0
        for i, (inputs, targets) in enumerate(loader_train):
            inputs = inputs.view([-1, dim_inp])
            targets_oh = F.one_hot(targets, num_classes=num_classes)
            out_pred = model.infer(targets_oh, inputs, K=K)

            optimizer.zero_grad()

            model.calc_updates()
            num_samples += inputs.shape[0]

            optimizer.step()

            model.clip_weights()

        # print(f"(Train) Avg Total discrepancy = {totd / (1.0 * num_samples)}, Avg BCE loss = {bce_loss / (1.0 * num_samples)}")

        val_ce_loss, val_acc = eval_model(model, loader_val, num_classes)

        if val_ce_loss < best_val_ce_loss:
            checkpoint_filename = f'{checkpoint_dir}/{trial_name}-model.pt'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename} (CE loss {best_val_ce_loss} -> {val_ce_loss})")
            best_val_ce_loss = val_ce_loss



if __name__ == '__main__':
    run_ngc(314159, trial_name='ngc-classify')