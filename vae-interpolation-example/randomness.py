import random

import numpy as np
from numpy.random import RandomState
import torch


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param reset_random_gen:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _ = RandomGenerator.get_instance(seed=seed)


def get_torch_generator(seed: int) -> torch.Generator:
    """
    Create a torch random generator independent of the global one, e.g., to
    shuffle a DataLoader in the same way regardless of how many random numbers
    were used before (for instance, to initialize the weights of a model).

    :param seed:
    """
    return torch.Generator().manual_seed(seed)


class RandomGenerator:
    __instance: "RandomGenerator" = None

    @staticmethod
    def get_instance(
        seed: int = 0,
    ) -> "RandomGenerator":
        if RandomGenerator.__instance is None:
            RandomGenerator(seed=seed)
        return RandomGenerator.__instance

    def __init__(self, seed: int):
        if RandomGenerator.__instance is not None:
            raise Exception("This class is a singleton!")

        self.rnd_state: RandomState = np.random.RandomState(seed=seed)
        RandomGenerator.__instance = self
