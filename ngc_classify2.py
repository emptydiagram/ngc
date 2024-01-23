from utils import init_gaussian, init_uniform, make_moving_collate_fn, make_lkwta, set_seed

from operator import itemgetter
import os
import pickle
import urllib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision

def calc_modulation(W):
    what = torch.sum(W, dim=1, keepdim=True)
    m = torch.minimum(2 * what / torch.max(what), torch.tensor(1.0))
    return m.repeat(1, W.shape[1])


class GNCN_Classify2:
    def __init__(self, config, device=None):
        # use itemgetter
        L, dims, fns_phi, weight_stddev, beta, error_update_factor = itemgetter(
            'L', 'dims', 'fns_phi', 'weight_stddev', 'beta', 'error_update_factor')(config)

        # we require that dims = [(dim_x, dim_y), dim_1, ..., dim_L]
        # and also that fns_phi = [phi^1, ..., phi^L]
        assert len(dims[0]) == 2, "dims[0] must be a tuple of (dim_inp, num_classes)"
        assert len(dims) == L + 1, "dims must have L+1 elements"
        assert len(fns_phi) == L, "fns_phi must have L elements"

        self.L = L
        self.dims = dims
        self.dims_x, self.dims_y = dims[0]
        self.fns_phi = [None] + list(map(self.get_activation_fn, fns_phi)) # pad to make indexing slightly less confusing
        self.beta = beta
        self.error_update_factor = error_update_factor

        self.device = torch.device('cpu') if device is None else device

        self.W1 = [init_gaussian([dims[1], self.dims_x], weight_stddev, self.device), init_gaussian([dims[1], self.dims_y], weight_stddev, self.device)]
        self.W2_L = [init_gaussian([dims[i+1], dims[i]], weight_stddev, self.device) for i in range(1, L)]
        self.E1 = [init_gaussian([self.dims_x, dims[1]], weight_stddev, self.device), init_gaussian([self.dims_y, dims[1]], weight_stddev, self.device)]
        self.E2_L = [init_gaussian([dims[i], dims[i+1]], weight_stddev, self.device) for i in range(1, L)]

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
        return self.W1 + self.W2_L + self.E1 + self.E2_L

    def state_dict(self):
        state = {}
        state[f'W{1}x'] = self.W1[0]
        state[f'W{1}y'] = self.W1[1]
        state[f'E{1}x'] = self.E1[0]
        state[f'E{1}y'] = self.E1[1]
        for l in range(self.L - 1):
            state[f'W{l+2}'] = self.W2_L[l]
            state[f'E{l+2}'] = self.E2_L[l]
        return state

    def load_state_dict(self, state):
        self.W1[0] = state[f'W{1}x']
        self.W1[1] = state[f'W{1}y']
        self.E1[0] = state[f'E{1}x']
        self.E1[1] = state[f'E{1}y']
        for l in range(self.L - 1):
            self.W2_L[l] = state[f'W{l+2}']
            self.E2_L[l] = state[f'E{l+2}']


    def infer(self, x, y=None, K=50):
        batch_size = x.shape[0]
        no_y = y is None
        if no_y:
            y = torch.zeros([batch_size, self.dims_y], device=self.device)

        z = [[x, y]]
        # e = [(torch.zeros([batch_size, self.dims_x], device=self.device), torch.zeros([batch_size, self.dims_y], device=self.device))]
        for i in range(1, self.L+1):
            z.append(torch.zeros([batch_size, self.dims[i]], device=self.device))
            # e.append(torch.zeros([batch_size, self.dims[i]], device=self.device))

        mu = [None for _ in range(self.L)]
        # create dummy e^L, initialized to zeros, which is never updated
        e = [None for _ in range(self.L)] + [torch.zeros([batch_size, self.dims[-1]], device=self.device)]
        # d[0] is padding
        d = [None for _ in range(self.L+1)]

        for _ in range(K):
            # compute predictions and prediction errors
            # mu^{L-1} = phi^L(z^L) W^L
            # e^{L-1} = z^{L-1} - mu^{L-1}
            # ...
            # mu^1 = phi^2(z^2) W^2
            # e^1 = z^1 - mu^1
            # mu^0_x = phi^1(z^1) W^1_x
            # mu^0_y = phi^1(z^1) W^1_y
            # e^0_x = z^0_x - mu^0_x
            # e^0_y = z^0_y - mu^0_y
            for l in range(self.L-1, 0, -1):
                mu[l] = self.fns_phi[l+1](z[l+1]) @ self.W2_L[l-1]
                e[l] = z[l] - mu[l]
            
            mu[0] = (self.fns_phi[1](z[1]) @ self.W1[0], self.fns_phi[1](z[1]) @ self.W1[1])
            e[0] = (z[0][0] - mu[0][0], z[0][1] - mu[0][1])

            # compute state updates
            # d^L = - e^L + e^{L-1} E^L 
            # z^L = z^L + beta * d^L
            # ...
            # d^2 = - e^2 + e^1 E^2
            # z^2 = z^2 + beta * d^2
            # d^1 = - e^1 + e^0_x E^1_x + e^0_y E^1_y
            # z^1 = z^1 + beta * d^1
            # if predicting:
            #   d^0_y = - e^0_y
            #   z^0_y = z^0_y + beta * d^0_y
            for l in range(self.L, 1, -1):
                d[l] = -e[l] + e[l-1] @ self.E2_L[l-2]
                z[l] = z[l] + self.beta * d[l]
            d[1] = -e[1] + e[0][0] @ self.E1[0] + e[0][1] @ self.E1[1]
            z[1] = z[1] + self.beta * d[1]

            if no_y:
                d0_y = -e[0][1]
                z[0][1] = z[0][1] + self.beta * d0_y


        self.z = z
        self.e = e
        self.d = d
        return mu[0][1]

    def calc_updates(self):
        batch_size = self.z[0][0].shape[0]
        avg_factor = -1.0 / (batch_size)

        # dW^L = (phi^L(z^L)^T e^{L-1}) * S_W^L
        # dE^L = gamma * (e^{L-1} d^L) * S_E^L
        # ...
        # dW^2 = (phi^2(z^2)^T e^1) * S_W^2
        # dE^2 = gamma * (e^1 d^2) * S_E^2
        # dW^1_x = (phi^1(z^1)^T e^0_x) * S_{W,x}^1
        # dW^1_y = (phi^1(z^1)^T e^0_y) * S_{W,y}^1
        # dE^1_x = gamma * (e^0_x d^1) * S_{E,x}^1
        # dE^1_y = gamma * (e^0_y d^1) * S_{E,y}^1

        for l in range(self.L-1, 0, -1):
            self.W2_L[l-1].grad = avg_factor * (self.fns_phi[l+1](self.z[l+1]).T @ self.e[l]) * calc_modulation(self.W2_L[l-1])
            self.E2_L[l-1].grad = avg_factor * (self.error_update_factor * self.e[l].T @ self.d[l+1]) * calc_modulation(self.E2_L[l-1])
        self.W1[0].grad = avg_factor * (self.fns_phi[1](self.z[1]).T @ self.e[0][0]) * calc_modulation(self.W1[0])
        self.W1[1].grad = avg_factor * (self.fns_phi[1](self.z[1]).T @ self.e[0][1]) * calc_modulation(self.W1[1])
        self.E1[0].grad = avg_factor * (self.error_update_factor * self.e[0][0].T @ self.d[1]) * calc_modulation(self.E1[0])
        self.E1[1].grad = avg_factor * (self.error_update_factor * self.e[0][1].T @ self.d[1]) * calc_modulation(self.E1[1])







def preprocess_mnist(batch_size, device, N_per_class=None):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)

    moving_collate = make_moving_collate_fn(device)

    if N_per_class is not None:
        selected_idxs = []
        for label in range(10):
            label_idxs = torch.where(data_train.targets == label)[0]
            label_selected_idxs = label_idxs[torch.randperm(len(label_idxs))[:N_per_class]]
            selected_idxs.extend(label_selected_idxs.tolist())

        data_train = torch.utils.data.Subset(data_train, selected_idxs)

    # split into train and validation
    if N_per_class is None:
        train_size, val_size = 50000, 10000
    else:
        val_size = int(0.15 * len(data_train))
        train_size = len(data_train) - val_size

    data_train, data_val = random_split(data_train, [train_size, val_size])

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

def eval_model(model, loader, num_classes, infer_K):
    num_samples = 0
    tot_ce_loss = 0.
    tot_correct = 0
    for (inputs, targets) in loader:
        inputs = inputs.view([-1, model.dims_x])
        preds = model.infer(inputs, None, K=infer_K)
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

    num_epochs = 20
    num_classes = 10
    batch_size = 500
    lr = 0.001
    dim_inp = 784
    dim_hid = 360
    K = 30
    N_per_class = None

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    loader_train, loader_val = preprocess_mnist(batch_size, device, N_per_class=N_per_class)
    # loader_train, loader_val = preprocess_mnist1d(batch_size, device)

    ngc_config = {
        'L': 3,
        'dims': [(dim_inp, num_classes), dim_hid, dim_hid, dim_hid],
        'fns_phi': ['tanh', 'tanh', 'tanh'],
        'weight_stddev': 0.025,
        'beta': 0.05,
        'error_update_factor': 0.98,
    }

    model = GNCN_Classify2(ngc_config, device=device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, maximize=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, maximize=False)

    val_ce_loss, val_acc = eval_model(model, loader_val, num_classes, K)
    best_val_ce_loss = val_ce_loss

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch}")
        num_samples = 0
        for i, (inputs, targets) in enumerate(loader_train):
            inputs = inputs.view([-1, dim_inp])
            targets_oh = F.one_hot(targets, num_classes=num_classes)
            out_pred = model.infer(inputs, targets_oh, K)

            optimizer.zero_grad()

            model.calc_updates()
            num_samples += inputs.shape[0]

            optimizer.step()


        # print(f"(Train) Avg Total discrepancy = {totd / (1.0 * num_samples)}, Avg BCE loss = {bce_loss / (1.0 * num_samples)}")

        val_ce_loss, val_acc = eval_model(model, loader_val, num_classes, K)

        if val_ce_loss < best_val_ce_loss:
            checkpoint_filename = f'{checkpoint_dir}/{trial_name}-model.pt'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename} (CE loss {best_val_ce_loss} -> {val_ce_loss})")
            best_val_ce_loss = val_ce_loss



if __name__ == '__main__':
    run_ngc(314159, trial_name='ngc-classify')