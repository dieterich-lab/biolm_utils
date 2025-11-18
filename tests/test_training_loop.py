import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments
from transformers.data.data_collator import DefaultDataCollator

from biolm_utils.train_utils import compute_metrics_for_regression, get_trainer
from biolm_utils.trainer import RegressionTrainer


class DummyDataset(Dataset):
    def __init__(self, n=16):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # input_ids: (seq_len, features) - the trainer doesn't care as long as tensors are present
        # labels should be scalar tensor so that the data collator keeps the key 'labels'
        return {"input_ids": torch.randn(1, 10), "labels": torch.tensor(float(idx % 2))}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, input_ids, **kwargs):
        # input_ids is (batch, seq_len, features); reduce along seq (use last axis)
        if len(input_ids.shape) > 2:
            x = input_ids.squeeze(1)
        else:
            x = input_ids
        logits = self.lin(x.float())
        return {"logits": logits}


def test_minimal_training_loop(tmp_path):
    # Prepare dummy args
    args = Namespace()
    args.mode = "fine-tune"
    args.task = "regression"
    args.dev = False
    args.silent = True
    args.patience = 1
    args.batchsize = 2

    # Build tokenizer-less minimal environment
    train_ds = DummyDataset(n=8)
    val_ds = DummyDataset(n=4)

    model = DummyModel()

    targs = TrainingArguments(
        output_dir=str(tmp_path / "outputs"),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        disable_tqdm=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        remove_unused_columns=False,
            metric_for_best_model="mse",
            load_best_model_at_end=True,
    )
    # Ensure Trainer recognizes label column name for evaluation
    targs.label_names = ["labels"]

    # Compute metrics wrapper for Trainer's EvalPrediction -> (predictions, labels) tuple
    def compute_metrics(pred):
        fn = compute_metrics_for_regression(val_ds, tmp_path)
        return fn((pred.predictions, pred.label_ids))

    trainer = get_trainer(
        args,
        RegressionTrainer,
        model,
        None,
        targs,
        train_ds,
        val_ds,
        DefaultDataCollator(),
        compute_metrics,
        None,
    )

    # Check the trainer was created
    assert trainer is not None

    # Train 1 epoch - minimal smoke test
    trainer.train()

    # Check state present
    assert hasattr(trainer, "state")
    # Check that model forward still works after training
    sample = train_ds[0]
    out = model(**{"input_ids": sample["input_ids"].unsqueeze(0)})
    assert "logits" in out
