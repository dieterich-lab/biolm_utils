from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from biolm_utils.train_utils import (
    IdentityScaler,
    LogScaler,
    compute_metrics_for_classification,
    compute_metrics_for_regression,
)


class TestTrainUtils:
    def test_log_scaler(self):
        scaler = LogScaler()
        data = np.array([1, 2, 4])
        transformed = scaler.fit_transform(data)
        assert np.allclose(transformed, np.log(data))
        inverse = scaler.inverse_transform(transformed)
        assert np.allclose(inverse, data)

    def test_identity_scaler(self):
        scaler = IdentityScaler()
        data = np.array([1, 2, 4])
        transformed = scaler.fit_transform(data)
        assert np.allclose(transformed, data)
        inverse = scaler.inverse_transform(transformed)
        assert np.allclose(inverse, data)

    def test_compute_metrics_for_regression(self):
        # Mock dataset
        dataset = MagicMock()
        savepath = "/tmp/test"

        metrics_func = compute_metrics_for_regression(dataset, savepath)

        # Mock predictions as tuple (logits, labels)
        pred = (
            torch.tensor([[1.0], [2.0], [3.0]]),  # predictions
            torch.tensor([[1.1], [2.1], [2.9]]),  # labels
        )

        result = metrics_func(pred)
        assert "mse" in result
        assert "spearman rho" in result
        assert isinstance(result["mse"], float)
        assert isinstance(result["spearman rho"], float)

    def test_compute_metrics_for_classification(self):
        # Mock dataset
        dataset = MagicMock()
        dataset.LE.classes_ = np.array(["class_0", "class_1"])  # Mock label encoder
        savepath = Path("/tmp/test")
        savepath.mkdir(exist_ok=True)

        metrics_func = compute_metrics_for_classification(dataset, savepath)

        # Mock predictions as EvalPrediction-like object
        pred = MagicMock()
        pred.predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # logits
        pred.label_ids = torch.tensor([1, 0, 1])

        result = metrics_func(pred)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert isinstance(result["accuracy"], float)
