import time

import torch
from sklearn.model_selection import StratifiedKFold
from torch import tensor, nn
from torch.optim import Adam, Optimizer
from torch.nn.modules.loss import _Loss
from torch_geometric.loader import DataLoader

from utils.compute_device import COMPUTE_DEVICE

# set default dtype, as MPS Pytorch does not support float64
torch.set_default_dtype(torch.float32)


def cross_validation_with_val_set(dataset, model, loss_fn, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None) -> tuple[float, float, float]:

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(COMPUTE_DEVICE).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            if hasattr(model, 'fit'):
                loss = model.fit(train_loader, optimizer, loss_fn, COMPUTE_DEVICE)
            else:
                loss = fit(model, train_loader, optimizer, loss_fn, COMPUTE_DEVICE)
                
            val_losses.append(eval_loss(model, val_loader, loss_fn))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
                'wd': [param_group['lr'] for param_group in optimizer.param_groups],
            }

            print(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    # for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
    for _, idx in skf.split(torch.zeros(len(dataset)), [data.y for data in dataset]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def fit(
    model:nn.Module,
    train_loader:DataLoader,
    optimizer:Optimizer,
    loss_fn:_Loss,
    device:torch.device,
) -> float:
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(COMPUTE_DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += float(loss) * data.num_graphs
    
    return total_loss / len(train_loader.dataset)


# def train(model, optimizer, loader):
#     model.train()

#     total_loss = 0
#     for data in loader:
#         optimizer.zero_grad()
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         loss = F.nll_loss(out, data.y.view(-1))
#         loss.backward()
#         total_loss += loss.item() * num_graphs(data)
#         optimizer.step()
#     return total_loss / len(loader.dataset)


def eval_acc(model:nn.Module, loader:DataLoader) -> float:
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(COMPUTE_DEVICE)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model:nn.Module, loader:DataLoader, loss_fn:_Loss) -> float:
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(COMPUTE_DEVICE)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
        loss += loss_fn(out, data.y)
        # loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
