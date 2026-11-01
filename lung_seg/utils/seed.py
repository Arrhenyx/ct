import random

import numpy as np
import torch
from monai.utils import set_determinism


def seed_everything(seed: int = 42) -> None:
    set_determinism(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True