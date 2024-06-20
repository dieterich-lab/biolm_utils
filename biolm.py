import logging
import os
import random

import numpy as np
import torch
from cross_validation import cv_wrapper
from interpret import loo_scores
from train_tokenizer import tokenize
from train_utils import (
    compute_metrics_for_regression,
    create_reports,
    get_data,
    get_model,
    get_tokenizer,
    get_trainer,
)
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
    TrainerState,
    TrainingArguments,
)

from entry import *

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(0)
print(f"Random seed set as {0}")


if args.mode != "tokenize":
    TOKENIZER_CLS = TOKENIZERDICT[args.model]
    TOKENIZER = get_tokenizer(args, TOKENIZERFILE, TOKENIZER_CLS)
    TOKENIZER_FOR_TRAINER = (
        TOKENIZER
        if SPECIAL_TOKENIZER_FOR_TRAINER is None
        else SPECIAL_TOKENIZER_FOR_TRAINER
    )
    DATASET_CLS = DATASETDICT[args.model]
    DATASET = get_data(args, TOKENIZER, ADD_SPECIAL_TOKENS, DATASETFILE, DATASET_CLS)
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
    trainer_cls = MLMTRAINER if args.mode == "pre-train" else REGRESSIONTRAINER

    # Determine the output size of the network.
    if args.mode == "pre-train":
        nlabels = None
    else:  # regression tasks
        nlabels = 1

    # Getting the config.
    config = model_cls.get_config(
        args=args,
        config_cls=CONFIGCLS,
        tokenizer=tokenizer,
        dataset=DATASET,
        nlabels=nlabels,
    )

    # Getting the model.
    model = get_model(
        args=args,
        model_cls=model_cls,
        tokenizer=tokenizer,
        config=config,
        model_load_path=model_load_path,
        pretraining_required=PRETRAINING_REQUIRED,
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
        per_device_eval_batch_size=args.batchsize,
        gradient_accumulation_steps=GRADACC,
        save_total_limit=1,
        load_best_model_at_end=args.mode != "pre-train",
        evaluation_strategy="epoch" if args.mode != "pre-train" else "no",
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
        learning_rate=LEARNINGRATE,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        save_safetensors=False,
        report_to=["tensorboard"],
    )

    # Get the trainer for the training run.
    trainer = get_trainer(
        args,
        trainer_cls,
        model,
        TOKENIZER_FOR_TRAINER,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
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

    return model


def test(test_dataset, data_collator, model_load_path, model_cls=None, model=None):
    # Get the trainer class.
    trainer_cls = REGRESSIONTRAINER

    nlabels = 1

    # Load the pre-trained model if not given.
    if model is None:
        config = model_cls.get_config(
            args=args,
            config_cls=CONFIGCLS,
            tokenizer=TOKENIZER,
            dataset=DATASET,
            nlabels=nlabels,
        )
        model = get_model(
            args, model_cls, TOKENIZER, config, model_load_path, nlabels, test_dataset
        )

    # Define the test arguments.
    test_args = TrainingArguments(
        output_dir=OUTPUTPATH,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=args.batchsize,
        dataloader_drop_last=False,  # this should actually be set to False, but see issues below
        # dataloader_drop_last=True,  # weirdly, sometimes training crashes with this set to false
        log_level="info" if not args.silent else "critical",
        disable_tqdm=True,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        label_names=["labels"],
        save_safetensors=False,
    )

    # Define the trainer ("predictor") for the test set.
    evaluator = trainer_cls(
        model=model,
        tokenizer=TOKENIZER_FOR_TRAINER,
        args=test_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_regression,
    )

    # Get metrics and predictions from the test set.
    test_results = evaluator.predict(test_dataset)

    # Log/save test metrics and predictions from the test set.
    evaluator.log_metrics("test", test_results.metrics)
    evaluator.save_metrics("test", test_results.metrics)

    # Create the reports
    scaler = model.scaler
    create_reports(test_dataset, test_results, scaler, REPORTFILE, RANKFILE)

    if hasattr(test_dataset.dataset, "labels"):
        return test_results.metrics["test_spearman rho"]


@cv_wrapper(args, DATASET)
def run(train_dataset, val_dataset, test_dataset, model_load_path, model_save_path):

    if args.mode == "tokenize":
        tokenize(args)
    else:
        # Get the correct model class.
        if args.mode == "pre-train":
            model_cls = MODELDICT["pre-train"][args.model]
        else:
            model_cls = MODELDICT["fine-tune"][args.model]

        # Getting the corresponding data collator.
        if args.mode == "pre-train":
            if args.model == "xlnet":
                data_collator = DataCollatorForPermutationLanguageModeling(
                    tokenizer=TOKENIZER
                )
            else:
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=TOKENIZER, mlm=True, mlm_probability=0.15
                )
        elif args.mode in ["fine-tune", "predict"]:
            if args.model == "xlnet":
                data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
            else:
                data_collator = DefaultDataCollator()
        # Pre-training and fine-tuning.
        if args.mode in ["pre-train", "fine-tune"]:
            model = train(
                model_cls=model_cls,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
                model_save_path=model_save_path,
                tokenizer=TOKENIZER,
            )
            # Testing (inference) after cross-validation.
            if args.splitpos is not None:
                test_results = test(
                    test_dataset=test_dataset,
                    data_collator=data_collator,
                    model=model,
                    model_load_path=model_save_path,
                )
                return test_results
        # Testing (inference) an already trained model.
        elif args.mode == "predict":
            test_results = test(
                model_cls=model_cls,
                test_dataset=test_dataset,
                data_collator=data_collator,
                model_load_path=model_load_path,
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
                remove_first_last=ADD_SPECIAL_TOKENS,
            )
            return scores
        else:
            raise ValueError(f"Unknown mode: '{args.mode}'.")


if __name__ == "__main__":
    run  # Statement is enough as we didn't define a an actual wrapped function in `cross_validate`.
