import os
import random

import numpy as np
import pytest
import torch

from biolm_utils.biolm import set_seed


class TestBiolm:
    def test_set_seed(self):
        seed = 42
        set_seed(seed)
        val1 = random.random()
        val2 = np.random.random()
        val3 = torch.rand(1).item()

        set_seed(seed)
        assert random.random() == val1
        assert np.random.random() == val2
        assert torch.rand(1).item() == val3

        # Check environment
        assert os.environ["PYTHONHASHSEED"] == str(seed)
