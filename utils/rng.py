from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def preserve_rng_state(seed=None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
