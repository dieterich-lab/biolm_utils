import logging
import os
import pickle
import random

import numpy as np
import torch
from transformers.data.data_collator import DefaultDataCollator
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

from biolm_utils.config import get_config
from biolm_utils.cross_validation import parametrized_decorator
from biolm_utils.entry import (
    CHECKPOINTPATH,
    CLASSIFICATIONTRAINER_CLS,
    DATASETFILE,
    GRADACC,
    METRIC,
    MLMTRAINER_CLS,
    REGRESSIONTRAINER_CLS,
    TBPATH,
    TOKENIZERFILE,
    args,
)
from biolm_utils.interpret import loo_scores
from biolm_utils.train_tokenizer import tokenize
from biolm_utils.train_utils import (
    create_reports,
    get_dataset,
    get_model_and_config,
    get_tokenizer,
    get_trainer,
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
        return MLMTRAINER_CLS

    task_to_trainer = {
        "regression": REGRESSIONTRAINER_CLS,
        "classification": CLASSIFICATIONTRAINER_CLS,
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


def _build_training_args(model_save_path, val_dataset, config):
    """Builds the TrainingArguments for the main training loop."""
    eval_batch_size = args.batchsize
    if val_dataset and args.batchsize > len(val_dataset):
        eval_batch_size = len(val_dataset)

    is_pre_train = args.mode == "pre-train"
    load_best = not args.dev and not is_pre_train
    save_strategy = "epoch" if not args.dev else "no"
    eval_strategy = "epoch" if args.mode != "pre-train" else "no"

    num_epochs = int(args.resume) if not isinstance(args.resume, bool) else args.nepochs

    return TrainingArguments(
        output_dir=str(model_save_path),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=GRADACC,
        save_total_limit=1 if not args.dev else 0,
        load_best_model_at_end=load_best,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        logging_strategy="steps" if is_pre_train else "epoch",
        disable_tqdm=True,
        log_level="critical" if args.silent else "info",
        logging_dir=str(TBPATH),
        warmup_ratio=0.05 if is_pre_train else 0.0,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        label_names=["labels"],
        learning_rate=config.LEARNINGRATE,
        max_grad_norm=config.MAX_GRAD_NORM,
        weight_decay=config.WEIGHT_DECAY,
        save_safetensors=False,
        report_to=["tensorboard"],
    )


def _build_test_args(model_load_path, test_dataset):
    """Builds the TrainingArguments for testing/prediction."""
    if args.ngpus > 1:
        logging.warning(
            "Running inference on %d GPUs. This may drop samples if "
            "the dataset size is not divisible by the batch size. "
            "Consider using a single GPU for complete evaluation.",
            args.ngpus,
        )

    test_batch_size = min(args.batchsize, len(test_dataset))

    return TrainingArguments(
        output_dir=str(model_load_path),
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=test_batch_size,
        dataloader_drop_last=args.ngpus > 1,
        log_level="critical" if args.silent else "info",
        disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["labels"],
        save_safetensors=False,
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
    trainer_cls = _get_trainer_class(args.mode, args.task)
    num_labels = _get_num_labels(args.mode, args.task, full_dataset)

    model = get_model_and_config(
        args=args,
        model_cls=model_cls,
        model_config_cls=config.CONFIG_CLS,
        tokenizer=tokenizer,
        dataset=full_dataset,
        nlabels=num_labels,
        model_load_path=model_load_path,
        pretraining_required=config.PRETRAINING_REQUIRED,
        scaler=getattr(train_dataset.dataset, "scaler", None),
    )

    model_size = sum(p.numel() for p in model.parameters())
    logging.info(f"Model size: {model_size / 1e6:.1f}M parameters")

    training_args = _build_training_args(model_save_path, val_dataset, config)

    compute_metrics = (
        None if args.mode == "pre-train" else METRIC(full_dataset, model_save_path)
    )
    labels = getattr(full_dataset, "labels", None)

    trainer = get_trainer(
        args,
        trainer_cls,
        model,
        tokenizer_for_trainer,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        compute_metrics,
        labels,
    )

    num_epochs_trained = 0
    if args.resume is True:
        logging.info(f"Resuming training from checkpoint: {CHECKPOINTPATH}")
        train_result = trainer.train(resume_from_checkpoint=str(CHECKPOINTPATH))
    else:
        if not isinstance(args.resume, bool):
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

    trainer.save_model()
    tokenizer.save_pretrained(model_save_path)
    trainer.save_state()

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
        with open(model_save_path / "scaler.pkl", "wb") as f:
            pickle.dump(model.scaler, f)

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
    model = None,
):
    """Handles the model testing and prediction."""
    trainer_cls = _get_trainer_class(args.mode, args.task)

    if model is None:
        num_labels = _get_num_labels(args.mode, args.task, test_dataset.dataset)
        model = get_model_and_config(
            args=args,
            model_cls=model_cls,
            model_config_cls=config.CONFIG_CLS,
            tokenizer=tokenizer,
            dataset=full_dataset,
            nlabels=num_labels,
            model_load_path=model_load_path,
            pretraining_required=config.PRETRAINING_REQUIRED,
            scaler=None,
        )

    test_args = _build_test_args(model_load_path, test_dataset)
    compute_metrics = METRIC(full_dataset, model_load_path)
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
    config = get_config()

    if args.mode == "tokenize":
        tokenize(args)
        return
    print("TOKENIZERFILE", TOKENIZERFILE)
    stop
    # Initialize tokenizer and dataset, making them available for the `run` function.
    tokenizer = get_tokenizer(
        args, TOKENIZERFILE, config.TOKENIZER_CLS, config.PRETRAINING_REQUIRED
    )
    tokenizer_for_trainer = (
        tokenizer
        if config.SPECIAL_TOKENIZER_FOR_TRAINER_CLS is None
        else config.SPECIAL_TOKENIZER_FOR_TRAINER_CLS()
    )
    full_dataset = get_dataset(
        args, tokenizer, config.ADD_SPECIAL_TOKENS, DATASETFILE, config.DATASET_CLS
    )

    # By defining `run` inside `main`, the decorator can safely access `full_dataset`.
    @parametrized_decorator(args, full_dataset)
    def run(
        train_dataset,
        val_dataset,
        test_dataset,
        model_load_path,
        model_save_path,
        report_file,
        rank_file,
    ):
        """Main execution logic, called for each cross-validation fold."""

        model_cls_map = {
            "pre-train": config.MODEL_CLS_FOR_PRETRAINING,
            "fine-tune": config.MODEL_CLS_FOR_FINETUNING,
            "predict": config.MODEL_CLS_FOR_FINETUNING,
            "interpret": config.MODEL_CLS_FOR_FINETUNING,
        }
        model_cls = model_cls_map.get(args.mode)
        if model_cls is None:
            raise ValueError(f"Unknown mode: '{args.mode}'.")

        if args.mode == "pre-train":
            data_collator = config.DATACOLLATOR_CLS_FOR_PRETRAINING(tokenizer=tokenizer)
        else:  # fine-tune, predict, interpret
            data_collator = DefaultDataCollator()

        if args.mode in ["pre-train", "fine-tune"]:
            results, model = train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
                model_save_path=model_save_path,
                tokenizer=tokenizer,
                tokenizer_for_trainer=tokenizer_for_trainer,
                full_dataset=full_dataset,
                model_cls=model_cls,
                config=config,
            )
            if args.mode == "fine-tune" and test_dataset:
                results = test(
                    model=model,
                    test_dataset=test_dataset,
                    data_collator=data_collator,
                    model_load_path=model_save_path,
                    report_file=report_file,
                    rank_file=rank_file,
                    tokenizer=tokenizer,
                    tokenizer_for_trainer=tokenizer_for_trainer,
                    full_dataset=full_dataset,
                    model_cls=model_cls,
                    config=config,
                )
            return results

        elif args.mode == "predict":
            return test(
                test_dataset=test_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
                report_file=report_file,
                rank_file=rank_file,
                tokenizer=tokenizer,
                tokenizer_for_trainer=tokenizer_for_trainer,
                full_dataset=full_dataset,
                model_cls=model_cls,
                config=config,
            )

        elif args.mode == "interpret":
            return loo_scores(
                args=args,
                tokenizer=tokenizer,
                model_cls=model_cls,
                test_dataset=test_dataset,
                model_load_path=model_load_path,
                output_path=model_save_path,
                remove_first_last=config.ADD_SPECIAL_TOKENS,
            )

    # This call triggers the `parametrized_decorator`, which handles
    # cross-validation and calls the `run` function for each fold.
    run()


if __name__ == "__main__":
    set_seed(SEED)
    log_gpu_info()
    main()
