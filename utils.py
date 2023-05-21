import torch

# the the best compute device in a system
def GetDevice():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# get total model params
def TotalModelParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
