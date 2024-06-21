import json
import logging
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)


def compute_metrics_for_regression(pred):
    logits, labels = pred
    logits = logits.squeeze().tolist()
    labels = labels.squeeze().tolist()
    mse = mean_squared_error(labels, logits)
    spearman_rho, _ = spearmanr(logits, labels)
    return {
        "mse": mse,
        "spearman rho": spearman_rho,
    }


def get_tokenizer(args, tokenizer_file, tokenizer_cls):

    if args.mode != "pre-train" or args.pretrainedmodel:
        with open(tokenizer_file, "r") as f:
            tokenizer_json = json.load(f)
        # Remove the meta data left and right correctly
        # [1] and [2] refer to the position where the sequence is isolated by means of the `columnsep`
        tokenizer_json["pre_tokenizer"]["pretokenizers"][1]["pattern"][
            "Regex"
        ] = f"([^{args.columnsep}]*{args.columnsep}){{{int(args.seqpos) - 1}}}"
        tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"][
            "Regex"
        ] = f"{args.columnsep}.*"
        if args.tokensep is not None:
            tokenizer_json["normalizer"]["normalizers"][-2]["pattern"][
                "String"
            ] = args.tokensep
        # unfortunately we need to temporarily save the tokenizer as
        # XlnetTokenizerFast is deprived of the ability to load serialized tokenizers
        with tempfile.NamedTemporaryFile("r+") as tmp:
            json.dump(tokenizer_json, tmp)
            tmp.seek(0)
            tokenizer = tokenizer_cls(
                tokenizer_file=tmp.name,
                mask_token="[MASK]",
                cls_token="[CLS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                sep_token="[SEP]",
                bos_token="[BOS]",
                eos_token="[EOS]",
                model_max_length=args.blocksize,
                truncation=True,
                truncation_side="left" if args.lefttailing else "right",
            )
    else:  # pre-training data is the same as the data for tokenizing
        logger.info(f"Loading tokenizer from {tokenizer_file}")
        tokenizer = tokenizer_cls(
            tokenizer_file=str(tokenizer_file),
            mask_token="[MASK]",
            cls_token="[CLS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            sep_token="[SEP]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_max_length=args.blocksize,
            truncation=True,
            truncation_side="left" if args.lefttailing else "right",
        )
    tokenizer.name_or_path = tokenizer_file
    return tokenizer


def get_dataset(args, tokenizer, add_special_tokens, dataset_file, dataset_cls):
    if (
        not dataset_file.exists()  # required data file doesn't exist yet
        or args.getdata  # only tokenize the data and exit
        or args.dev  # debug mode
        or args.mode
        == "predict"  # here, we expect a new file which should not be saved
    ):
        dataset = dataset_cls(
            tokenizer=tokenizer,
            args=args,
            add_special_tokens=add_special_tokens,
        )
        if not args.dev and args.mode != "predict":
            logger.info(f"Saving dataset to {dataset_file}")
            with open(dataset_file, "wb") as f:
                pickle.dump(dataset, f)
        if args.getdata:
            sys.exit()
    else:  # dataset is available
        logger.info(f"Loading dataset from {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
        tokenizer = dataset.tokenizer
    # dataset.log_raw_data()
    # dataset.log_data()
    return dataset


def get_trainer(
    args,
    trainer_cls,
    model,
    tokenizer,
    training_args,
    train_dataset,
    val_dataset,
    data_collator,
):
    if args.mode == "pre-train":
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    else:  # fine-tuning tasks
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=(
                [
                    EarlyStoppingCallback(early_stopping_patience=args.patience),
                ]
                if not args.dev
                else None
            ),
            compute_metrics=compute_metrics_for_regression,
        )

    return trainer


def create_reports(test_dataset, test_results, scaler, report_path, rank_path):
    seqs = [test_dataset.dataset.seq_idx[i] for i in test_dataset.indices]

    # Get the results.
    preds = test_results.predictions.squeeze()

    # Transform predictions and gold labels to original space.
    preds = scaler.inverse_transform(np.array(preds).reshape(1, -1)).squeeze()

    # Save the sequence idx, predictions and true labels to a csv file.
    if hasattr(test_dataset.dataset, "labels"):
        labels = [test_dataset.dataset.labels[i] for i in test_dataset.indices]
        labels = scaler.inverse_transform(labels).squeeze()
        # Create a file with rank deltas.
        label_tups = list(enumerate(labels))
        label_seq_tups = [x + tuple([y]) for x, y in zip(label_tups, seqs)]
        label_seq_tups = sorted(label_seq_tups, key=lambda x: x[1])
        pred_tups = list(enumerate(preds))
        pred_tups = sorted(pred_tups, key=lambda x: x[1])
        label_ranks, sorted_labels, sorted_seqs = zip(*label_seq_tups)
        pred_ranks, sorted_preds = zip(*pred_tups)
        rank_deltas = [x - y for x, y in zip(label_ranks, pred_ranks)]
        rank_df = pd.DataFrame(
            list(
                zip(
                    sorted_seqs,
                    sorted_labels,
                    label_ranks,
                    sorted_preds,
                    pred_ranks,
                    rank_deltas,
                )
            ),
            columns=["seqs", "label", "label_rank", "pred", "pred_rank", "rank_delta"],
        )
        logger.info(f"Saving test rankings to {rank_path}.")
        rank_df.to_csv(rank_path, index=False)
        report_df = pd.DataFrame(
            list(zip(seqs, labels, preds)),
            columns=["sequence", "label", "prediction"],
        )
    else:
        report_df = pd.DataFrame(
            list(zip(seqs, preds)),
            columns=["sequence", "prediction"],
        )
    logger.info(f"Saving test predictions to {report_path}.")
    report_df.to_csv(report_path, index=False)


def get_model(
    args,
    model_cls,
    tokenizer,
    config,
    model_load_path,
    pretraining_required,
    scaler=None,
):
    if args.mode == "pre-train":
        if not args.resume:
            logger.info(f"Initializing new {model_cls} model for pre-training.")
        else:
            logger.info(
                f"Initializing new {model_cls} model for pre-training for later loading of pre-trained parameters."
            )
        model = model_cls(config=config)
        model.resize_token_embeddings(len(tokenizer))
    elif args.mode == "fine-tune":
        if pretraining_required:
            if (
                not args.fromscratch and args.resume == False
            ) or args.mode == "predict":
                try:
                    with open(Path(model_load_path) / "trainer_state.json") as f:
                        trainer_state = json.load(f)
                    n_epochs = trainer_state["log_history"][-1]["epoch"]
                except:
                    pass
                try:
                    n_epochs = trainer_state["epoch"]
                except:
                    n_epochs = "unknown"
                model = model_cls.from_pretrained(
                    model_load_path,
                    config=config,
                    use_safetensors=False,
                )
                logger.info(
                    f"Loaded {model_cls} model with weights from {model_load_path} saved on {datetime.fromtimestamp(model_load_path.stat().st_ctime)} with {n_epochs} epochs trained."
                )
            else:
                logger.info(f"Initializing new {model_cls} model for fine-tuning.")
                model = model_cls(config)
        else:
            model = model_cls(config)
        model.scaler = scaler
    elif args.mode in ["predict", "interpret"]:
        n_epochs = trainer_state["log_history"][-1]["epoch"]
        logger.info(
            f"Loaded {model_cls} model with weights from {model_load_path} saved on {datetime.fromtimestamp(model_load_path.stat().st_ctime)} with {n_epochs} epochs trained."
        )
        model = model_cls.from_pretrained(
            model_load_path,
            config=config,
            use_safetensors=False,
        )
    return model
