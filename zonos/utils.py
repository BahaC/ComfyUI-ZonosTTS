import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    # MPS breaks for whatever reason. Uncomment when it's working.
    # if torch.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


DEFAULT_DEVICE = get_device()
