import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights

from src import config


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def new_resnet(pretrained):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, 6)
    return model


def build_model(regime, seed, device):
    cfg = config.REGIMES[regime]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = new_resnet(cfg["pretrained"])
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        p.requires_grad = cfg["trainable"] is None or top in cfg["trainable"]
    return model.to(device)


def build_optimizer(model, regime):
    groups = []
    for modules, lr in config.REGIMES[regime]["groups"]:
        params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and (modules is None or n.split(".")[0] in modules)
        ]
        name = "all" if modules is None else "+".join(modules)
        groups.append({"params": params, "lr": lr, "name": name})
    return torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def set_train_mode(model, regime):
    model.train()
    for name in config.REGIMES[regime]["eval"]:
        getattr(model, name).eval()


def predict_logits(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()
