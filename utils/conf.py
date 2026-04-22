import random
import torch
import numpy as np
from monai.utils import set_determinism


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def data_path() -> str:
    return './data'


def base_path() -> str:
    return './data'


def checkpoint_path() -> str:
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed, use_deterministic_algorithms=False)
