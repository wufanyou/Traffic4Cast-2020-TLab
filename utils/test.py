import torch
import numpy as np
from tqdm import tqdm

__ALL__ = ["test_model", "test_model_ensemble", "test_model_ensembles"]


@torch.torch.no_grad()
def test_model(loader, model):
    model.eval()
    results = {}
    for i, data in tqdm(enumerate(loader)):
        idx = loader.dataset.sample[i]
        data = {k: v.float().cuda() for (k, v) in data.items()}
        output = model.logits(**data).detach()
        output = torch.clamp((output + 1) / 2 * 255, 0, 255).cpu().numpy()
        output = np.clip(np.round(output), 0, 255).astype(np.uint8)
        results[idx] = output
    return results


@torch.torch.no_grad()
def test_model_ensemble(loader, models, weight=None):
    for model in models:
        model.eval()
    results = {}
    weight = np.ones(len(models)) if weight is None else np.array(weight)
    weight = weight / (weight.sum(-1, keepdims=True))

    for i, data in tqdm(enumerate(loader)):
        idx = loader.dataset.sample[i]
        data = {k: v.float().cuda() for (k, v) in data.items()}
        x = 0
        for i, model in enumerate(models):
            x_ = model.logits(**data)
            x_ = torch.clamp((x_ + 1) / 2, 0, 1) * 255
            x += weight[i] * x_
        output = torch.clamp(x, 0, 255).cpu().numpy()
        output = output.astype(np.uint8)
        results[idx] = output
    return results


@torch.torch.no_grad()
def test_model_ensembles(loaders, models, weight=None):

    results = {}
    weight = np.ones(len(models)) if weight is None else np.array(weight)
    weight = weight / (weight.sum(-1, keepdims=True))

    for model in models:
        model.eval()

    iter_loaders = [iter(loader) for loader in loaders]
    for id in tqdm(range(len(loaders[0]))):
        idx = loaders[0].dataset.sample[id]
        all_data = [next(iter_loader) for iter_loader in iter_loaders]
        x = 0
        for i, model in enumerate(models):
            data = {k: v.cuda() for (k, v) in all_data[i].items()}
            x_ = model.logits(**data)
            x_ = torch.clamp((x_ + 1) / 2, 0, 1) * 255
            x += weight[i] * x_
        output = torch.clamp(x, 0, 255).cpu().numpy()
        output = np.clip(np.round(output), 0, 255).astype(np.uint8)
        results[idx] = output
    return results
