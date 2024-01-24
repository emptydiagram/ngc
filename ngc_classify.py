from utils import make_moving_collate_fn, set_seed
from models import GNCN_PDH_Classify, NGC_ANGC

import os
import pickle
import urllib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision




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



def run_ngc_pdh(seed, trial_name='ngc'):
    set_seed(seed)

    num_epochs = 200
    num_classes = 10
    batch_size = 500
    lr = 0.001
    dim_inp = 784
    dim_hid = 360
    K = 60
    # err_update_coeff = 0.95
    N_per_class = None

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    loader_train, loader_val = preprocess_mnist(batch_size, device, N_per_class=N_per_class)
    # loader_train, loader_val = preprocess_mnist1d(batch_size, device)

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

def run_ngc_angc(seed, trial_name='ngc'):
    set_seed(seed)

    num_epochs = 200
    num_classes = 10
    batch_size = 500
    lr = 0.001
    dim_inp = 784
    dim_hid = 360
    K = 60
    # err_update_coeff = 0.95
    N_per_class = None

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    loader_train, loader_val = preprocess_mnist(batch_size, device, N_per_class=N_per_class)
    # loader_train, loader_val = preprocess_mnist1d(batch_size, device)

    ngc_config = {
        'L': 3,
        'dims': [num_classes, dim_hid, dim_hid, dim_inp],
        'fns_phi': ['identity', 'relu', 'relu', 'relu', 'identity'],
        'weight_stddev': 0.025,
        'beta': 0.1,
        'beta_e': 0.5,
        'leak': 0.001,
    }
    model = NGC_ANGC(ngc_config['L'], ngc_config['dims'], ngc_config['weight_stddev'], beta=ngc_config['beta'], beta_e=ngc_config['beta_e'], gamma=ngc_config['leak'], device=device)

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

            model.normalize_weights()

        # print(f"(Train) Avg Total discrepancy = {totd / (1.0 * num_samples)}, Avg BCE loss = {bce_loss / (1.0 * num_samples)}")

        val_ce_loss, val_acc = eval_model(model, loader_val, num_classes)

        if val_ce_loss < best_val_ce_loss:
            checkpoint_filename = f'{checkpoint_dir}/{trial_name}-model.pt'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename} (CE loss {best_val_ce_loss} -> {val_ce_loss})")
            best_val_ce_loss = val_ce_loss



if __name__ == '__main__':
    # run_ngc_pdh(314159, trial_name='ngc-pdh-classify')
    run_ngc_angc(314159, trial_name='ngc-angc')