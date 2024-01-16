from utils import init_gaussian_dense, make_moving_collate_fn, set_seed

import os

import torch
import torch.nn.functional as F
import torchvision

class NGC_Classify:
    def __init__(self, L, dim_top, dim_hid, dim_bot, weight_stddev, beta=0.1, gamma=0.001, err_update_coeff=1.0, alpha_m=0,
                 fn_phi_name='relu', fn_g_hid_name='identity', fn_g_out_name='softmax', device=None):
        self.L = L
        self.dim_top = dim_top
        self.dim_hid = dim_hid
        self.dim_bot = dim_bot
        self.beta = beta
        self.gamma = gamma # leak coefficient
        self.err_update_coeff = err_update_coeff
        self.alpha_m = alpha_m

        self.device = torch.device('cpu') if device is None else device

        self.W = ([init_gaussian_dense([dim_hid, dim_bot], weight_stddev, self.device)]
            + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-2)]
            + [init_gaussian_dense([dim_top, dim_hid], weight_stddev, self.device)])

        self.E = ([init_gaussian_dense([dim_bot, dim_hid], weight_stddev, self.device)]
            + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-2)]
            + [init_gaussian_dense([dim_hid, dim_top], weight_stddev, self.device)])

        self.M = []
        if self.alpha_m > 0:
            self.M = ([init_gaussian_dense([dim_hid, dim_bot], weight_stddev, self.device)]
                + [init_gaussian_dense([dim_hid, dim_hid], weight_stddev, self.device) for _ in range(L-3)]
                + [init_gaussian_dense([dim_top, dim_hid], weight_stddev, self.device)])


        if fn_phi_name == 'relu':
            self.fn_phi = torch.relu
        elif fn_phi_name == 'sigmoid':
            self.fn_phi = torch.sigmoid
        else:
            raise NotImplementedError("Only relu, sigmoid are supported for phi.")

        self.fn_phi_top = lambda x: x

        if fn_g_hid_name == 'identity':
            self.fn_g_hid = lambda x: x
        else:
            raise NotImplementedError("Only identity is supported for g_hid.")

        if fn_g_out_name == 'softmax':
            self.fn_g_out = lambda x: torch.softmax(x, dim=1)
        else:
            raise NotImplementedError("Only softmax is supported for g_out.")

        self.clip_weights()


    def parameters(self):
        return self.W + self.E

    def state_dict(self):
        state = {}
        for l in range(self.L):
            state[f'W{l}'] = self.W[l]
            state[f'E{l}'] = self.E[l]
        return state

    def load_state_dict(self, state):
        for l in range(self.L):
            self.W[l] = state[f'W{l}']
            self.E[l] = state[f'E{l}']

    def project(self, x_top):
        zbar = x_top
        zbar = self.fn_g_hid(self.fn_phi_top(zbar) @ self.W[self.L - 1])
        for l in range(self.L - 2, 0, -1):
            zbar = self.fn_g_hid(self.fn_phi(zbar) @ self.W[l])

        zbar = self.fn_g_out(self.fn_phi(zbar) @ self.W[0])
        return zbar


    def infer(self, x_bot, x_top, K=50):
        batch_size = x_bot.shape[0]
        z = [x_bot]
        e = [torch.zeros([batch_size, self.dim_bot], device=self.device)]
        for l in range(self.L - 1):
            z.append(torch.zeros([batch_size, self.dim_hid], device=self.device))
            e.append(torch.zeros([batch_size, self.dim_hid], device=self.device))
        z.append(x_top)
        e.append(torch.zeros([batch_size, self.dim_top], device=self.device))

        mu = [None for _ in range(self.L)]

        for _ in range(K):
            for i in range(1, self.L):
                di = e[i-1] @ self.E[i-1] - e[i]
                z[i] += self.beta * (-self.gamma * z[i] + di)

            mu_W_input = self.fn_phi(z[1]) @ self.W[0]
            mu[0] = self.fn_g_out(mu_W_input)
            e[0] = z[0] - mu[0]

            for i in range(1, self.L - 1):
                mu_W_input = self.fn_phi(z[i+1]) @ self.W[i]
                mu[i] = self.fn_g_hid(mu_W_input)
                e[i] = self.fn_phi(z[i]) - mu[i]

            mu_W_input = self.fn_phi_top(z[self.L]) @ self.W[self.L - 1]
            mu[self.L - 1] = self.fn_g_hid(mu_W_input)
            e[self.L - 1] = self.fn_phi(z[self.L - 1]) - mu[self.L - 1]

        self.z = z
        self.e = e

        return mu[0]

    def calc_updates(self):
        batch_size = self.z[0].shape[0]
        avg_factor = -1.0 / (batch_size)

        for l in range(0, self.L - 1):
            dWl = self.fn_phi(self.z[l+1]).T @ self.e[l]
            dWl = avg_factor * dWl
            dEl = dWl.T
            self.W[l].grad = dWl
            self.E[l].grad = dEl

        dWl = self.fn_phi_top(self.z[self.L]).T @ self.e[self.L - 1]
        dWl = avg_factor * dWl
        dEl = dWl.T
        self.W[self.L - 1].grad = dWl
        self.E[self.L - 1].grad = dEl

    def clip_weights(self):
        # clip column norms to 1
        for l in range(self.L):
            Wl_col_norms = self.W[l].norm(dim=0, keepdim=True)
            self.W[l].copy_(self.W[l] / torch.maximum(Wl_col_norms, torch.tensor(1.0)))
            El_col_norms = self.E[l].norm(dim=0, keepdim=True)
            self.E[l].copy_(self.E[l] / torch.maximum(El_col_norms, torch.tensor(1.0)))



def preprocess_mnist(batch_size, device):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)

    moving_collate = make_moving_collate_fn(device)

    # split into train and validation
    data_train, data_val = torch.utils.data.random_split(data_train, [50000, 10000])

    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=moving_collate)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, collate_fn=moving_collate)
    return loader_train, loader_val


def cross_entropy_loss(targets, predictions, eps=1e-7):
    clamped_predictions = torch.clamp(predictions, min=eps, max=1.0 - eps)
    return -torch.sum(targets * torch.log(clamped_predictions))

def eval_model(model, loader, num_classes):
    num_samples = 0
    tot_ce_loss = 0.
    tot_correct = 0
    for (inputs, targets) in loader:
        inputs = inputs.view([-1, model.dim_top])
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
    batch_size = 512
    lr = 0.001
    dim_inp = 784
    dim_hid = 360
    weight_stddev = 0.05
    L = 3
    K = 50
    beta = 0.1
    gamma = 0.001

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    loader_train, loader_val = preprocess_mnist(batch_size, device)

    model = NGC_Classify(L=L, dim_top=dim_inp, dim_hid=dim_hid, dim_bot=num_classes, weight_stddev=weight_stddev, beta=beta, gamma=gamma, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=False)

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