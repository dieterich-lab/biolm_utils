import json
import logging
import os
import pickle
import random
import warnings

import numpy as np
import torch
from transformers.data.data_collator import DefaultDataCollator
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)

from .config_access import ConfigManager
from .constants import get_constants
from .cross_validation import CrossValidator
from .path_setup import PathsManager
from .plugin_config import PluginManager

args = None
constants = None
paths = None
from .interpret import loo_scores
from .params import get_detected_ngpus
from .paths import Paths
from .runner import make_run_fn
from .train_tokenizer import tokenize
from .train_utils import (
    create_reports,
    get_dataset,
    get_model_and_config,
    get_tokenizer,
    get_trainer,
)


def initialize_runtime(config, log_params: bool = False):
    """Propagate Hydra config to module-level singletons and caches."""
    global args, constants, paths

    ConfigManager._instance = config
    PathsManager.set_config(config)

    # Refresh derived caches with the new config
    from .constants import get_constants as _get_constants
    from .constants import reset_constants

    reset_constants()
    args = config
    constants = _get_constants(args=config, log_params=log_params)
    paths = PathsManager.get_paths(config=config)


def ensure_runtime(config=None, *, log_params: bool = False):
    """Ensure global runtime state is initialized, using provided or default config."""

    global args

    if config is not None:
        if args is None or args is not config:
            initialize_runtime(config, log_params=log_params)
        return

    if args is not None:
        return

    if ConfigManager._instance is not None:
        initialize_runtime(ConfigManager._instance, log_params=log_params)
        return

    raise RuntimeError(
        "Runtime is not initialized. Call initialize_runtime(config) before invoking training or testing utilities."
    )


# --- Configuration & Setup ---

SEED = 0


def set_seed(seed):
    """Sets the seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def log_gpu_info():
    """Logs information about available GPUs."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logging.info(f"GPU available: True. Number of devices: {device_count}.")
    else:
        logging.info("GPU available: False.")


def _get_trainer_class(mode, task):
    """Determines the appropriate Trainer class based on mode and task."""
    if mode == "pre-train":
        return constants["MLMTRAINER_CLS"]

    task_to_trainer = {
        "regression": constants["REGRESSIONTRAINER_CLS"],
        "classification": constants["CLASSIFICATIONTRAINER_CLS"],
    }
    trainer_cls = task_to_trainer.get(task)
    if trainer_cls is None:
        raise ValueError(f"Invalid task '{task}' for mode '{mode}'.")
    return trainer_cls


def _get_num_labels(mode, task, dataset):
    """Determines the number of output labels for the model."""
    if mode == "pre-train":
        return None
    if task == "classification":
        return dataset.LE.classes_.size
    return 1  # For regression tasks


def _build_training_args(model_save_path, val_dataset, config, train_dataset):
    """Builds the TrainingArguments for the main training loop."""
    eval_batch_size = args.training.batchsize
    if val_dataset and args.training.batchsize > len(val_dataset):
        eval_batch_size = len(val_dataset)

    is_pre_train = args.mode == "pre-train"
    load_best = not args.debugging.dev and not is_pre_train
    save_strategy = "epoch" if not args.debugging.dev else "no"
    eval_strategy = "epoch" if args.mode != "pre-train" else "no"

    num_epochs = (
        int(args.training.resume)
        if not isinstance(args.training.resume, bool)
        else args.training.nepochs
    )

    # Calculate logging_steps for ~10 logs per epoch
    detected_gpus = get_detected_ngpus(args)
    effective_batch_size = (
        args.training.batchsize * constants["GRADACC"] * detected_gpus
    )
    steps_per_epoch = len(train_dataset) // effective_batch_size
    logging_steps = max(1, steps_per_epoch // 10)
    logging.info(
        f"Calculated logging_steps={logging_steps} for ~10 logs per epoch (steps_per_epoch={steps_per_epoch})"
    )

    return TrainingArguments(
        output_dir=str(model_save_path),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.training.batchsize,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=constants["GRADACC"],
        save_total_limit=1 if not args.debugging.dev else 0,
        load_best_model_at_end=load_best,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy="steps",
        logging_steps=logging_steps,
        disable_tqdm=True,
        log_level="info",
        warmup_ratio=0.05 if is_pre_train else 0.0,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        label_names=["labels"],
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        save_safetensors=True,
        report_to=[],
    )


def _build_test_args(model_load_path, test_dataset):
    """Builds the TrainingArguments for testing/prediction."""
    detected_gpus = get_detected_ngpus(args)
    if detected_gpus > 1:
        logging.warning(
            "Running inference on %d GPUs. This may drop samples if "
            "the dataset size is not divisible by the batch size. "
            "Consider using a single GPU for complete evaluation.",
            detected_gpus,
        )

    test_batch_size = min(args.training.batchsize, len(test_dataset))

    return TrainingArguments(
        output_dir=str(model_load_path),
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=test_batch_size,
        dataloader_drop_last=detected_gpus > 1,
        log_level="info",
        disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["labels"],
        save_safetensors=True,
    )


# --- Core Training and Evaluation Functions ---


def train(
    train_dataset,
    val_dataset,
    data_collator,
    model_load_path,
    model_save_path,
    tokenizer,
    tokenizer_for_trainer,
    full_dataset,
    model_cls,
    config,
):
    """Handles the model training loop."""
    ensure_runtime()
    trainer_cls = _get_trainer_class(args.mode, args.task)
    num_labels = _get_num_labels(args.mode, args.task, full_dataset)

    model = get_model_and_config(
        args=args,
        model_cls=model_cls,
        model_config_cls=config.config_cls,
        tokenizer=tokenizer,
        dataset=full_dataset,
        nlabels=num_labels,
        model_load_path=model_load_path,
        pretraining_required=config.pretraining_required,
        scaler=getattr(train_dataset.dataset, "scaler", None),
    )

    model_size = sum(p.numel() for p in model.parameters())
    logging.info(f"Model size: {model_size / 1e6:.1f}M parameters")

    training_args = _build_training_args(
        model_save_path, val_dataset, config, train_dataset
    )

    compute_metrics = (
        None
        if args.mode == "pre-train"
        else constants["METRIC"](full_dataset, model_save_path)
    )
    labels = getattr(full_dataset, "labels", None)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer_for_trainer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    num_epochs_trained = 0
    if args.training and args.training.resume is True:
        logging.info(
            f"Resuming training from checkpoint: {constants['CHECKPOINTPATH']}"
        )
        train_result = trainer.train(
            resume_from_checkpoint=str(constants["CHECKPOINTPATH"])
        )
    else:
        if not isinstance(args.training.resume, bool):
            trainer._load_from_checkpoint(model_save_path)
            state_path = model_save_path / "trainer_state.json"
            trainer.state = TrainerState.load_from_json(state_path)
            num_epochs_trained = trainer.state.epoch
            logging.info(
                f"Loaded trainer state with {num_epochs_trained:.2f} epochs trained."
            )
        train_result = trainer.train()

    logging.info(f"Saving best model to {model_save_path}")
    if num_epochs_trained > 0:
        trainer.state.num_train_epochs += num_epochs_trained

    model.scaling_method = getattr(args, "scaling", "identity")  # Save scaling method
    model.config.scaling_method = model.scaling_method  # Save in config

    # Save the model with scaling metadata
    trainer.save_model()
    tokenizer.save_pretrained(model_save_path)
    trainer.save_state()

    # --- Copy best checkpoint to final_model directory ---
    import re
    import shutil
    from pathlib import Path

    checkpoint_dirs = [
        d
        for d in model_save_path.iterdir()
        if d.is_dir() and re.match(r"checkpoint-\d+", d.name)
    ]
    if checkpoint_dirs:
        # Sort by checkpoint number
        checkpoint_dirs.sort(key=lambda d: int(d.name.split("-")[-1]))
        best_ckpt = checkpoint_dirs[-1]
        final_model_dir = model_save_path / "final_model"
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_ckpt, final_model_dir)
        logging.info(f"Copied best checkpoint {best_ckpt} to {final_model_dir}")

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    if args.mode == "pre-train":
        try:
            train_metrics["perplexity"] = np.exp(train_metrics["train_loss"])
        except OverflowError:
            train_metrics["perplexity"] = float("inf")
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    eval_metrics = {}
    if args.mode != "pre-train":
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(val_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    metric_key = {
        "classification": "eval_f1",
        "regression": "eval_spearman rho",
    }.get(args.task)

    return eval_metrics.get(metric_key, 0.0), model


def test(
    test_dataset,
    data_collator,
    model_load_path,
    report_file,
    rank_file,
    tokenizer,
    tokenizer_for_trainer,
    full_dataset,
    model_cls,
    config,
    model,
):
    """Handles the model testing and prediction."""
    ensure_runtime()
    trainer_cls = _get_trainer_class(args.mode, args.task)

    if model is None:
        num_labels = _get_num_labels(args.mode, args.task, test_dataset.dataset)
        model = get_model_and_config(
            args=args,
            model_cls=model_cls,
            model_config_cls=config.config_cls,
            tokenizer=tokenizer,
            dataset=full_dataset,
            nlabels=num_labels,
            model_load_path=model_load_path,
            pretraining_required=config.pretraining_required,
            scaler=None,
        )
    # Set scaler and scaling_method from the dataset
    if hasattr(full_dataset, "scaler") and full_dataset.scaler is not None:
        model.scaler = full_dataset.scaler
    if hasattr(full_dataset, "scaling_method"):
        model.scaling_method = full_dataset.scaling_method
    if hasattr(model, "scaling_method") and model.scaling_method:
        logging.info(f"Model uses scaling method: {model.scaling_method}")
    print("="*100, flush=True)
    print(args, flush=True)
    print("="*100, flush=True)

    test_args = _build_test_args(model_load_path, test_dataset)
    compute_metrics = constants["METRIC"](full_dataset, model_load_path)
    labels = getattr(full_dataset, "labels", None)
    evaluator = get_trainer(
        args,
        trainer_cls,
        model,
        tokenizer_for_trainer,
        test_args,
        None,
        None,
        data_collator,
        compute_metrics,
        labels,
    )

    test_results = evaluator.predict(test_dataset)
    evaluator.log_metrics("test", test_results.metrics)
    evaluator.save_metrics("test", test_results.metrics)

    create_reports(test_dataset, test_results, model.scaler, report_file, rank_file)

    metric_key = {
        "regression": "test_spearman rho",
        "classification": "test_f1",
    }.get(args.task)

    return test_results.metrics.get(metric_key, 0.0)


# --- Main Dispatcher ---


def main():
    """Main execution entry point."""
    # Set up global logging
    logging.basicConfig(
        level=logging.INFO,  # Set global logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set up logging using the centralized configuration
    from .logging_config import setup_logging

    setup_logging(paths.get("LOGFILE"))

    config = PluginManager.get_config()

    if args.mode == "tokenize":
        tokenize(args)
        return

    # Initialize tokenizer and dataset, making them available for the `run` function.
    tokenizer = get_tokenizer(
        args, paths["TOKENIZERFILE"], config.tokenizer_cls, config.pretraining_required
    )
    tokenizer_for_trainer = (
        tokenizer
        if config.special_tokenizer_for_trainer_cls is None
        else config.special_tokenizer_for_trainer_cls()
    )
    full_dataset = get_dataset(
        args,
        tokenizer,
        config.add_special_tokens,
        paths["DATASETFILE"],
        config.dataset_cls,
    )

    # Build a run-once function and hand orchestration to the CrossValidator
    # NOTE: we intentionally keep the per-fold function signature identical to
    # the previous nested `run` function so the CrossValidator may invoke it
    # without further changes.

    run_once = make_run_fn(
        args=args,
        config=config,
        tokenizer=tokenizer,
        tokenizer_for_trainer=tokenizer_for_trainer,
        full_dataset=full_dataset,
    )

    base_paths = Paths(
        model_load_path=paths["MODELLOADPATH"],
        model_save_path=paths["MODELSAVEPATH"],
        output_path=paths["OUTPUTPATH"],
        report_file=paths["REPORTFILE"],
        rank_file=paths["RANKFILE"],
    )

    cv = CrossValidator(
        params=args, dataset=full_dataset, run_once_fn=run_once, base_paths=base_paths
    )
    return cv.execute()


if __name__ == "__main__":
    set_seed(SEED)
    log_gpu_info()
    main()
