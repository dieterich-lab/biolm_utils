import logging
import os
import random

import numpy as np
import torch
from transformers import DefaultDataCollator, TrainerState, TrainingArguments

from biolm_utils.config import get_config
from biolm_utils.cross_validation import parametrized_decorator
from biolm_utils.entry import (  # DATASET_CLS,
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

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(0)
logging.info(f"Random seed set as {0}")
logging.info(
    f"GPU available: {torch.cuda.is_available()}. "
    + f"Number of devices: {torch.cuda.device_count()}."
    if torch.cuda.device_count()
    else ""
)


config = get_config()

if args.mode != "tokenize":
    TOKENIZER = get_tokenizer(args, TOKENIZERFILE, config.TOKENIZER_CLS)
    TOKENIZER_FOR_TRAINER = (
        TOKENIZER
        if config.SPECIAL_TOKENIZER_FOR_TRAINER_CLS is None
        else config.SPECIAL_TOKENIZER_FOR_TRAINER_CLS()
    )
    DATASET = get_dataset(
        args, TOKENIZER, config.ADD_SPECIAL_TOKENS, DATASETFILE, config.DATASET_CLS
    )
else:
    DATASET = None
    TOKENIZER = None


def train(
    model_cls,
    train_dataset,
    val_dataset,
    data_collator,
    model_load_path,
    model_save_path,
    tokenizer,
):

    # Get the trainer class.
    trainer_cls = (
        MLMTRAINER_CLS
        if args.mode == "pre-train"
        else (
            REGRESSIONTRAINER_CLS
            if args.task == "regression"
            else CLASSIFICATIONTRAINER_CLS
        )
    )

    # Determine the output size of the network.
    if args.mode == "pre-train":
        nlabels = None
    else:  # regression tasks
        nlabels = DATASET.LE.classes_.size if args.task == "classification" else 1

    # Getting the model.
    model = get_model_and_config(
        args=args,
        model_cls=model_cls,
        model_config_cls=config.CONFIG_CLS,
        tokenizer=TOKENIZER,
        dataset=DATASET,
        nlabels=nlabels,
        # model_config=model_config,
        model_load_path=model_load_path,
        pretraining_required=config.PRETRAINING_REQUIRED,
        scaler=train_dataset.dataset.scaler,
    )

    # Log the total of trainable parameters.
    model_size = sum(t.numel() for t in model.parameters())
    logging.info(f"Model size: {model_size/1000**2:.1f}M parameters")

    training_args = TrainingArguments(
        output_dir=model_save_path,
        overwrite_output_dir=True,
        num_train_epochs=(
            args.nepochs if isinstance(args.resume, bool) else int(args.resume)
        ),
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=(
            args.batchsize
            if val_dataset is None or args.batchsize < len(val_dataset)
            else len(val_dataset)
        ),
        gradient_accumulation_steps=GRADACC,
        save_total_limit=1,
        load_best_model_at_end=args.mode != "pre-train",
        eval_strategy="epoch" if args.mode != "pre-train" else "no",
        save_strategy="epoch",
        logging_strategy="epoch" if args.mode != "pre-train" else "steps",
        disable_tqdm=True,
        log_level="info" if not args.silent else "critical",
        logging_dir=TBPATH,
        warmup_ratio=0.05 if args.mode == "pre-train" else 0,
        remove_unused_columns=False,
        dataloader_drop_last=True,  # for training, we want to avoid issues with `batchnorm1d`
        ignore_data_skip=False,
        label_names=["labels"],
        learning_rate=config.LEARNINGRATE,
        max_grad_norm=config.MAX_GRAD_NORM,
        weight_decay=config.WEIGHT_DECAY,
        save_safetensors=False,
        report_to=["tensorboard"],
    )

    # Get the trainer for the training run.
    COMPUTE_METRICS = (
        None if args.mode == "pre-train" else METRIC(DATASET, model_save_path)
    )
    try:
        labels = DATASET.labels
    except:
        labels = None

    trainer = get_trainer(
        args,
        trainer_cls,
        model,
        TOKENIZER_FOR_TRAINER,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        COMPUTE_METRICS,
        labels,
    )

    if args.resume == True:
        # Finishing off interrupted training.
        train_result = trainer.train(resume_from_checkpoint=CHECKPOINTPATH)
    else:
        # Further pre-training.
        if not isinstance(args.resume, bool):
            trainer._load_from_checkpoint(model_save_path)
            trainer.state = TrainerState.load_from_json(
                model_save_path / "trainer_state.json"
            )
            num_epochs_trained = trainer.state.epoch
            logging.info(
                f"Loaded trainer state with {num_epochs_trained} epochs trained."
            )
        train_result = trainer.train()

    # Save the trained model and training metrics.
    logging.info(f"Saving best model to {model_save_path}")
    if not isinstance(args.resume, bool):
        trainer.state.num_train_epochs += num_epochs_trained
    trainer.save_model()
    # We save the tokenizer additionally in the model path for referencing during inference with
    # a model from a different experiment.
    tokenizer.save_pretrained(model_save_path)
    train_metrics = train_result.metrics
    if args.mode == "pre-train":
        try:
            perplexity = np.exp(train_metrics["train_loss"])
        except OverflowError:
            perplexity = float("inf")
        train_metrics["perplexity"] = perplexity
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    # Get validation metrics and add the size of the validation set.
    if args.mode != "pre-train":
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(val_dataset)

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        if hasattr(train_dataset.dataset, "labels"):
            if args.task == "classification":
                return eval_metrics["eval_f1"]
            else:
                return eval_metrics["eval_spearman rho"]


def test(
    test_dataset,
    data_collator,
    model_load_path,
    report_file,
    rank_file,
    model_cls=None,
    model=None,
):
    # Get the trainer class.
    trainer_cls = (
        REGRESSIONTRAINER_CLS
        if args.task == "regression"
        else CLASSIFICATIONTRAINER_CLS
    )

    nlabels = 1 if args.task == "regression" else test_dataset.dataset.LE.classes_.size

    # Load the pre-trained model if not given.
    if model is None:
        model = get_model_and_config(
            args=args,
            model_cls=model_cls,
            model_config_cls=config.CONFIG_CLS,
            tokenizer=TOKENIZER,
            dataset=DATASET,
            nlabels=nlabels,
            model_load_path=model_load_path,
            pretraining_required=config.PRETRAINING_REQUIRED,
            scaler=test_dataset.dataset.scaler,
        )

    # Define the test arguments.
    if args.ngpus > 1:
        logging.info(
            f"""
            Warning: You are training on more than one GPU ({args.ngpus}).
            This may lead to batches being dropped when the size of your dataset is not divisable by the batch size.
            To make ensure that all your samples are being predicted, we advise that you carry out inference on a single GPU.
            """
        )
    test_args = TrainingArguments(
        output_dir=model_load_path,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=(
            args.batchsize if args.batchsize < len(test_dataset) else len(test_dataset)
        ),
        dataloader_drop_last=True if args.ngpus > 1 else False,
        log_level="info" if not args.silent else "critical",
        disable_tqdm=True,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        label_names=["labels"],
        save_safetensors=False,
    )

    # Define the trainer ("predictor") for the test set.
    COMPUTE_METRICS = METRIC(DATASET, model_load_path)
    labels = DATASET.labels
    evaluator = get_trainer(
        args,
        trainer_cls,
        model,
        TOKENIZER_FOR_TRAINER,
        test_args,
        None,
        None,
        data_collator,
        COMPUTE_METRICS,
        labels,
    )

    # Get metrics and predictions from the test set.
    test_results = evaluator.predict(test_dataset)

    # Log/save test metrics and predictions from the test set.
    evaluator.log_metrics("test", test_results.metrics)
    evaluator.save_metrics("test", test_results.metrics)

    # Create the reports
    try:
        scaler = model.scaler
    except:
        scaler = test_dataset.dataset.scaler
    create_reports(test_dataset, test_results, scaler, report_file, rank_file)

    # if hasattr(test_dataset.dataset, "labels"):
    if args.task == "regression":
        return test_results.metrics["test_spearman rho"]
    if args.task == "classification":
        return test_results.metrics["test_f1"]


@parametrized_decorator(args, DATASET)
def run(
    train_dataset,
    val_dataset,
    test_dataset,
    model_load_path,
    model_save_path,
    report_file=None,
    rank_file=None,
    output_path=None,
):

    if args.mode == "tokenize":
        tokenize(args)
    else:
        # Get the model class.
        if args.mode == "pre-train":
            model_cls = config.MODEL_CLS_FOR_PRETRAINING
        else:
            model_cls = config.MODEL_CLS_FOR_FINETUNING

        # Getting the corresponding data collator.
        if args.mode == "pre-train":
            data_collator = config.DATACOLLATOR_CLS_FOR_PRETRAINING(tokenizer=TOKENIZER)
        elif args.mode in ["fine-tune", "predict"]:
            data_collator = DefaultDataCollator()
        # Pre-training and fine-tuning.
        if args.mode in ["pre-train", "fine-tune"]:
            eval_results = train(
                model_cls=model_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
                model_save_path=model_save_path,
                tokenizer=TOKENIZER,
            )
            if (
                args.mode == "pre-train"
                or args.splitpos == None
                or args.splitpos == False
            ):
                return eval_results
            else:
                test_results = test(
                    model_cls=model_cls,
                    test_dataset=test_dataset,
                    data_collator=data_collator,
                    model_load_path=model_save_path,
                    report_file=report_file,
                    rank_file=rank_file,
                )
            return test_results
        # Testing (inference) an already trained model or test on the splits.
        # elif args.mode in ["fine-tune", "predict"]:
        elif args.mode == "predict":
            test_results = test(
                model_cls=model_cls,
                test_dataset=test_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
                report_file=report_file,
                rank_file=rank_file,
            )
            return test_results
        # Calculation of LOO scores for a trained model.
        elif args.mode == "interpret":
            scores = loo_scores(
                args=args,
                tokenizer=TOKENIZER,
                model_cls=model_cls,
                test_dataset=test_dataset,
                model_load_path=model_load_path,
                output_path=output_path,
                remove_first_last=config.ADD_SPECIAL_TOKENS,
            )
            return scores
        else:
            raise ValueError(f"Unknown mode: '{args.mode}'.")


if __name__ == "__main__":
    run()
