from utils import init_gaussian_dense, make_moving_collate_fn, set_seed

import os

import torch
import torchvision


class GNCN_PDH:
    def __init__(self, L, dim_top, dim_hid, dim_inp, weight_stddev, beta=0.1, gamma=0.001, alpha_m=0, fn_phi_name='relu', fn_g_hid_name='relu', fn_g_out_name='sigmoid', device=None):
        self.L = L
        self.dim_top = dim_top
        self.dim_hid = dim_hid
        self.dim_inp = dim_inp
        self.beta = beta
        self.gamma = gamma # leak coefficient
        self.alpha_m = alpha_m

        self.device = torch.device('cpu') if device is None else device

        self.W = ([init_gaussian_dense([dim_hid, dim_inp], weight_stddev, self.device)]
            + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-2)]
            + [init_gaussian_dense([dim_top, dim_hid], weight_stddev, self.device)])

        self.E = ([init_gaussian_dense([dim_inp, dim_hid], weight_stddev, self.device)]
            + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-1)])

        # M3 is (dim_top, dim_hid)
        # M2 is (dim_hid, dim_inp)
        self.M = []
        if self.alpha_m > 0:
            self.M = ([init_gaussian_dense([dim_hid, dim_inp], weight_stddev, self.device)]
                + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-3)]
                + [init_gaussian_dense([dim_top, dim_hid], weight_stddev, self.device)])


        if fn_phi_name == 'relu':
            self.fn_phi = torch.relu
        else:
            raise NotImplementedError("Only relu is supported for phi.")

        if fn_g_hid_name == 'relu':
            self.fn_g_hid = torch.relu
        else:
            raise NotImplementedError("Only relu is supported for g_hid.")

        if fn_g_out_name == 'sigmoid':
            self.fn_g_out = torch.sigmoid
        else:
            raise NotImplementedError("Only sigmoid is supported for g_out.")

        self.clip_weights()


    def parameters(self):
        return self.W + self.E + self.M

    def state_dict(self):
        state = {}
        for l in range(self.L):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        if self.alpha_m == 1:
            for l in range(self.L - 1):
                state[f'M{l}'] = self.M[l]
        return state

    def load_state_dict(self, state):
        for l in range(self.L):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']
        if self.alpha_m == 1:
            for l in range(self.L - 1):
                state[f'M{l}'] = self.M[l]


    def infer(self, x, K=50):
        batch_size = x.shape[0]
        z = [x]
        e = [torch.zeros([batch_size, self.dim_inp], device=self.device)]
        for l in range(self.L - 1):
            z.append(torch.zeros([batch_size, self.dim_hid], device=self.device))
            e.append(torch.zeros([batch_size, self.dim_hid], device=self.device))
        z.append(torch.zeros([batch_size, self.dim_top], device=self.device))
        e.append(torch.zeros([batch_size, self.dim_top], device=self.device))

        mu = [None for _ in range(self.L)]

        for _ in range(K):
            for i in range(1, self.L + 1):
                di = e[i-1] @ self.E[i-1] - e[i]
                z[i] += self.beta * (-self.gamma * z[i] + di)

            mu_W_input = self.fn_phi(z[1]) @ self.W[0]
            if self.alpha_m == 1:
                mu[0] = self.fn_g_out(mu_W_input + self.fn_phi(z[2]) @ self.M[0])
            else:
                mu[0] = self.fn_g_out(mu_W_input)
            e[0] = z[0] - mu[0]
            for i in range(1, self.L):
                mu_W_input = self.fn_phi(z[i+1]) @ self.W[i]
                if self.alpha_m == 1 and i < self.L - 1:
                    mu[i] = self.fn_g_hid(mu_W_input + self.fn_phi(z[i+2]) @ self.M[i])
                else:
                    mu[i] = self.fn_g_hid(mu_W_input)
                e[i] = self.fn_phi(z[i]) - mu[i]

        self.z = z
        self.e = e

        return mu[0]

    def calc_updates(self):
        batch_size = self.z[0].shape[0]
        avg_factor = -1.0 / (batch_size)

        for l in range(0, self.L):
            dWl = self.fn_phi(self.z[l+1]).T @ self.e[l]
            dWl = avg_factor * dWl
            dEl = dWl.T
            self.W[l].grad = dWl
            self.E[l].grad = dEl

        if self.alpha_m == 1:
            for l in range(0, self.L - 1):
                dMl = self.fn_phi(self.z[l+2]).T @ self.e[l]
                dMl = avg_factor * dMl
                self.M[l].grad = dMl

    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))


    def calc_total_discrepancy(self):
        return sum([torch.sum(e**2) for e in self.e[:3]])



class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, data):
        return (data >= self.threshold).float()



def preprocess_binary_mnist(batch_size, device):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), Binarize()])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)

    moving_collate = make_moving_collate_fn(device)

    # split into train and validation
    data_train, data_val = torch.utils.data.random_split(data_train, [50000, 10000])

    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=moving_collate)
    return loader_train, loader_val

def binary_cross_entropy(targets, predictions, eps=1e-7):
    clamped_predictions = torch.clamp(predictions, min=eps, max=1.0 - eps)
    return -torch.sum(targets * torch.log(clamped_predictions) + (1.0 - targets) * torch.log(1.0 - clamped_predictions))

def eval_model(model, loader):
    num_samples = 0
    tot_discrep = 0.
    bce_loss = 0.
    for (inputs, _targets) in loader:
        inputs = inputs.view([-1, model.dim_inp])
        out_pred = model.infer(inputs, K=50)
        tot_discrep += model.calc_total_discrepancy()
        bce_loss += binary_cross_entropy(inputs, out_pred)
        num_samples += inputs.shape[0]
    avg_discrep = tot_discrep / (1.0 * num_samples)
    avg_bce_loss = bce_loss / (1.0 * num_samples)
    print(f"(Eval)  Avg Total discrepancy = {avg_discrep}, Avg BCE loss: {avg_bce_loss}")
    return avg_discrep, avg_bce_loss


def run_ngc(seed, trial_name='ngc'):
    set_seed(seed)

    num_epochs = 50
    batch_size = 512
    lr = 0.001
    dim_inp = 784
    dim_hid = 360
    weight_stddev = 0.05
    L = 3
    K = 50
    beta = 0.1
    gamma = 0.001
    alpha_m = 1

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    loader_train, loader_val = preprocess_binary_mnist(batch_size, device)

    model = GNCN_PDH(L=L, dim_top=dim_hid, dim_hid=dim_hid, dim_inp=dim_inp, weight_stddev=weight_stddev, beta=beta, gamma=gamma, alpha_m=alpha_m, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=False)

    val_discrep, val_bce_loss = eval_model(model, loader_val)
    best_val_bce_loss = val_bce_loss

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch}")
        totd = 0.
        bce_loss = 0.
        num_samples = 0
        for i, (inputs, _targets) in enumerate(loader_train):
            inputs = inputs.view([-1, dim_inp])
            out_pred = model.infer(inputs, K=K)

            optimizer.zero_grad()

            model.calc_updates()
            totd += model.calc_total_discrepancy()
            bce_loss += binary_cross_entropy(inputs, out_pred)
            num_samples += inputs.shape[0]

            optimizer.step()

            model.clip_weights()

        print(f"(Train) Avg Total discrepancy = {totd / (1.0 * num_samples)}, Avg BCE loss = {bce_loss / (1.0 * num_samples)}")

        val_discrep, val_bce_loss = eval_model(model, loader_val)

        if val_bce_loss < best_val_bce_loss:
            checkpoint_filename = f'{checkpoint_dir}/{trial_name}-model.pt'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename} (BCE loss {best_val_bce_loss} -> {val_bce_loss})")
            best_val_bce_loss = val_bce_loss



if __name__ == '__main__':
    run_ngc(314159, trial_name='base-ngc-skip')