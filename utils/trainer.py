import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

__ALL__ = [
    "train_model",
    "valid_model",
    "valid_model_ensemble",
    "valid_model_ensembles",
    "valid_model_ensemble_zeros",
    "valid_model_ensembles_geometric_mean",
    "valid_model_ensembles_zeros",
]


def train_model(step, epoch, loader, model, optimizer, scheduler=None, writer=None):
    model.train()
    writer.reset("train", epoch)
    for data in loader:
        step += 1
        optimizer.zero_grad()
        data = {k: v.cuda() for (k, v) in data.items()}
        # print(type(model))
        loss = model(**data)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss = loss.detach().cpu().item()
        writer.update(loss, data["x"].size(0), step)
    writer.write()
    return step


@torch.torch.no_grad()
def valid_model(step, epoch, loader, model, writer, mask=None):
    model.eval()
    writer.reset("valid", epoch)
    for data in loader:
        step += 1
        data = {k: v.cuda() for (k, v) in data.items()}
        if mask is not None:
            data["mask"] = mask
        loss = model(**data)
        loss = loss.detach().cpu().item()
        writer.update(loss, data["x"].size(0), step)
    writer.write()
    return step


@torch.torch.no_grad()
def valid_model_ensemble(loader, models, weights=None):
    losses = defaultdict(lambda: [])
    predict = []
    weights = np.ones(len(models)) if weights is None else np.array(weights)
    weights = weights / (weights.sum(-1, keepdims=True))

    if len(weights.shape) == 1:
        weights = weights[None]
    for model in models:
        model.eval()
    for data in tqdm(loader):
        predict = []
        data = {k: v.float().cuda() for (k, v) in data.items()}
        for i, model in enumerate(models):
            x = model.logits(**data)
            x = (
                torch.clamp(torch.round(torch.clamp((x + 1) / 2, 0, 1) * 255), 0, 255)
                / 255
            )
            predict.append(x.detach().clone())

        y = data["y"] / 255
        for i, x in enumerate(predict):
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[i].append(loss)

        for i, weight in enumerate(weights):
            x = 0
            for i, _x in enumerate(predict):
                # print(_x)
                x += weight[i] * _x
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[str(weight)].append(loss)

    return losses


@torch.torch.no_grad()
def valid_model_ensembles(loaders, models, weights=None):
    losses = defaultdict(lambda: [])
    predict = []
    weights = np.ones(len(models)) if weights is None else np.array(weights)
    weights = weights / (weights.sum(-1, keepdims=True))

    if len(weights.shape) == 1:
        weights = weights[None]

    for model in models:
        model.eval()

    iter_loaders = [iter(loader) for loader in loaders]
    for i in tqdm(range(len(loaders[0]))):
        all_data = [next(iter_loader) for iter_loader in iter_loaders]
        predict = []
        for i, model in enumerate(models):
            data = {k: v.float().cuda() for (k, v) in all_data[i].items()}
            x = model.logits(**data)
            x = (
                torch.clamp(torch.round(torch.clamp((x + 1) / 2, 0, 1) * 255), 0, 255)
                / 255
            )
            predict.append(x.detach().clone())
            y = data["y"] / 255
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[i].append(loss)

        for weight in weights:
            x = 0
            for i, _x in enumerate(predict):
                x += weight[i] * _x
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[str(weight)].append(loss)

    return losses


@torch.torch.no_grad()
def valid_model_ensemble_zeros(
    loader, zero_loader, model, zero_model, thresholds=[0.9]
):
    losses = defaultdict(lambda: [])
    predict = []
    model.eval()
    zero_model.eval()

    iter_loader = iter(loader)
    iter_zero_loader = iter(zero_loader)

    for i in tqdm(range(len(loader))):
        data = {k: v.float().cuda() for (k, v) in next(iter_loader).items()}
        x = model.logits(**data)
        x = torch.clamp(torch.round(torch.clamp((x + 1) / 2, 0, 1) * 255), 0, 255) / 255
        y = data["y"] / 255
        # loss = F.mse_loss(x, y)
        # loss = loss.detach().cpu().item()
        # losses['0'].append(loss)
        zero_data = {k: v.float().cuda() for (k, v) in next(iter_zero_loader).items()}
        mask = zero_model.logits(**zero_data)
        mask = torch.sigmoid(mask)

        raw_x = x.detach().clone()
        for t in thresholds:
            x = raw_x.clone()
            x[mask < t] = 0
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[str(t)].append(loss)

    return losses


@torch.torch.no_grad()
def valid_model_ensembles_geometric_mean(loaders, models, weights=None):
    losses = defaultdict(lambda: [])
    predict = []
    weights = np.ones(len(models)) if weights is None else np.array(weights)
    weights = weights / (weights.sum(-1, keepdims=True))

    if len(weights.shape) == 1:
        weights = weights[None]

    for model in models:
        model.eval()

    iter_loaders = [iter(loader) for loader in loaders]
    for i in tqdm(range(len(loaders[0]))):
        all_data = [next(iter_loader) for iter_loader in iter_loaders]
        predict = []
        for i, model in enumerate(models):
            data = {k: v.float().cuda() for (k, v) in all_data[i].items()}
            x = model.logits(**data)
            x = (
                torch.clamp(torch.round(torch.clamp((x + 1) / 2, 0, 1) * 255), 0, 255)
                / 255
            )
            predict.append(x.detach().clone())
            y = data["y"] / 255
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[i].append(loss)

        for weight in weights:
            x = 0
            for i, _x in enumerate(predict):
                x += weight[i] * torch.log(_x * 255 + 1)
            x = torch.exp(x)
            x = torch.clamp(torch.round(x - 1), 0, 255) / 255
            loss = F.mse_loss(x, y)
            loss = loss.detach().cpu().item()
            losses[str(weight)].append(loss)
    return losses


@torch.torch.no_grad()
def valid_model_ensembles_zeros(loaders, models, weight=[0.5,0.5], zero_threshold=range(1,11)):
    losses = defaultdict(lambda: [])
    for model in models:
        model.eval()

    iter_loaders = [iter(loader) for loader in loaders]
    for i in tqdm(range(len(loaders[0]))):
        all_data = [next(iter_loader) for iter_loader in iter_loaders]
        predict = []
        for i, model in enumerate(models):
            data = {k: v.float().cuda() for (k, v) in all_data[i].items()}
            x = model.logits(**data)
            x = (x + 1) / 2
            predict.append(x.detach().clone())

        y = data["y"] / 255
        x = 0
        for i, _x in enumerate(predict):
            x += weight[i] * _x
        for threshold in zero_threshold:
            z = x.clone()
            z = torch.clamp(z, 0, 1)
            z[z<threshold/10/255] = 0
            loss = F.mse_loss(z, y)
            loss = loss.detach().cpu().item()
            losses[str(threshold)].append(loss)
    return losses