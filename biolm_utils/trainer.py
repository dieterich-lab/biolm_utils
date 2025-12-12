import torch
import torch.nn as nn
from transformers import Trainer


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        targets = inputs.pop("labels")
        inputs.pop("qualities", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        targets = targets.type(logits.dtype)
        loss = torch.nn.functional.mse_loss(logits.squeeze(), targets.squeeze())
        return (loss, outputs) if return_outputs else loss


class WeightedRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        targets = inputs.pop("labels")
        qualities = inputs.pop("qualities")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(
            logits.squeeze(), targets.squeeze(), reduction="none"
        )
        loss = torch.mean(qualities * loss)
        return (loss, outputs) if return_outputs else loss


class WeightedSamplingTrainer(Trainer):
    def __init__(self, weights, **args):
        self.weights = weights
        super().__init__(**args)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.weights.to(self.args.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
