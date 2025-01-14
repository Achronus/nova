import torch


def set_device(device: str) -> torch.device:
    """Configures the `torch.device` based on the device `string` input."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)
