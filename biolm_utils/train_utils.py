import json
import logging
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.utils import class_weight
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)


class LogScaler:
    def fit_transform(self, data):
        return np.log(data)

    def inverse_transform(self, data):
        return np.exp(data)


# Not pretty but complies best with the rest of the code.
class IdentityScaler:
    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def compute_metrics_for_regression(dataset, savepath):
    def _compute_metrics(pred):
        logits, labels = pred
        logits = logits.squeeze().tolist()
        labels = labels.squeeze().tolist()
        mse = mean_squared_error(labels, logits)
        spearman_rho, _ = spearmanr(logits, labels)
        return {
            "mse": mse,
            "spearman rho": spearman_rho,
        }

    return _compute_metrics


def compute_metrics_for_classification(dataset, savepath):
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)
        # target_names = [dataset.LE.classes_[x] for x in names]
        target_names = dataset.LE.classes_.tolist()
        # used_labels = list(set(preds).union(set(labels)))
        used_labels = list(range(len(target_names)))
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            target_names=target_names,
            labels=used_labels,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(savepath / "classification_report.csv")
        logging.info(
            classification_report(
                labels,
                preds,
                target_names=target_names,
                labels=used_labels,
                zero_division=0,
            )
        )
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    return _compute_metrics


def get_tokenizer(args, tokenizer_file, tokenizer_cls, pretraining_required):

    if args.mode == "pre-train" or (
        args.mode == "fine-tune" and not pretraining_required
    ):
        mask_token = "[MASK]"
        pad_token = "[PAD]"
        cls_token = "[CLS]"
        unk_token = "[UNK]"
        sep_token = "[SEP]"
        eos_token = "[EOS]"
        bos_token = "[BOS]"
        trunc_side = "left" if not args.lefttailing else "right"
        model_max_len = args.blocksize

    else:
        if pretraining_required:
            tokenizer_config_file = (
                tokenizer_file.parent / "pre-train" / "tokenizer_config.json"
            )
        else:
            tokenizer_config_file = tokenizer_file.parent / "tokenizer_config.json"
        print(tokenizer_file.parent)
        # if pretraining_required:
        with open(
            tokenizer_config_file,
            "r",
        ) as ff:
            tok_config = json.load(ff)
            trunc_side = tok_config["truncation_side"]
            model_max_len = tok_config["model_max_length"]
            cls_token = tok_config["cls_token"]
            unk_token = tok_config["unk_token"]
            mask_token = tok_config["mask_token"]
            pad_token = tok_config["pad_token"]
            sep_token = tok_config["sep_token"]
            eos_token = tok_config["eos_token"]
            bos_token = tok_config["bos_token"]
        logger.info(
            f"Loaded tokenizer config from {tokenizer_file.parent / 'tokenizer_config.json'} and setting it to {model_max_len} model max length"
        )
        # Remove the meta data left and right correctly
        # [1] and [2] refer to the position where the sequence is isolated by means of the `columnsep`

    with open(tokenizer_file, "r") as f:
        tokenizer_json = json.load(f)
    print(tokenizer_file)
    print("-"*50)
    print(tok_config)
    tokenizer_json["pre_tokenizer"]["pretokenizers"][1]["pattern"][
        "Regex"
    ] = f"^([^{args.columnsep}]*{args.columnsep}){{{int(args.seqpos) - 1}}}"
    tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"][
        "Regex"
    ] = f"{args.columnsep}.*"
    if args.tokensep is not None:
        # Last position (-1) is for stripping the quotation marks.
        # We need to include the new tokensep replacement before.
        num_elements = len(tokenizer_json["normalizer"]["normalizers"])
        if (
            num_elements > 1
        ):  # this means a previous replacement with args.tokensep exists
            tokenizer_json["normalizer"]["normalizers"][-2]["pattern"][
                "String"
            ] = args.tokensep
        else:  # here, we have to create a new one
            if args.encoding == "bpe":
                replacement = ""
            elif args.encoding == "atomic":
                replacement = " "
            pattern = (
                {
                    "type": "Replace",
                    "pattern": {"String": args.tokensep},
                    "content": replacement,
                },
            )
            tokenizer_json["normalizer"]["normalizers"].insert(0, pattern)
    # unfortunately we need to temporarily save the tokenizer as
    # some instances of TokenizerFast are deprived of the ability to load serialized tokenizers
    with tempfile.NamedTemporaryFile("r+") as tmp:
        json.dump(tokenizer_json, tmp)
        tmp.seek(0)
        tokenizer = tokenizer_cls(
            tokenizer_file=tmp.name,
            mask_token=mask_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            bos_token=bos_token,
            eos_token=eos_token,
            model_max_length=model_max_len,
            truncation=True,
            truncation_side=trunc_side,
        )
    # else:  # pre-training data is the same as the data for tokenizing
    #     logger.info(
    #         f"Loading tokenizer from {tokenizer_file} and setting it to {args.blocksize} model max length"
    #     )
    #     tokenizer = tokenizer_cls(
    #         tokenizer_file=str(tokenizer_file),
    #         mask_token="[MASK]",
    #         cls_token="[CLS]",
    #         unk_token="[UNK]",
    #         pad_token="[PAD]",
    #         sep_token="[SEP]",
    #         bos_token="[BOS]",
    #         eos_token="[EOS]",
    #         model_max_length=args.blocksize,
    #         truncation=True,
    #         truncation_side="left" if args.lefttailing else "right",
    #     )
    tokenizer.name_or_path = tokenizer_file
    return tokenizer


def get_dataset(args, tokenizer, add_special_tokens, dataset_file, dataset_cls):
    print(dataset_file)
    if (
        not dataset_file.exists()  # required data file doesn't exist yet
        or args.getdata  # only tokenize the data and exit
        or args.dev  # debug mode
        or args.forcenewdata  # debug mode
        # or args.mode
        # == "predict"  # here, we expect a new file which should not be saved
    ):
        dataset = dataset_cls(
            tokenizer=tokenizer,
            args=args,
            add_special_tokens=add_special_tokens,
        )
        # if args.mode != "predict" and not args.dev:
        if not args.dev:
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
    compute_metrics,
    labels,
):
    if args.mode == "pre-train":
        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # labels=None,
        )
    elif args.task == "regression":
        # else:  # fine-tuning tasks
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
            compute_metrics=compute_metrics,
        )
    elif args.task == "classification":
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(labels), y=np.array(labels)
        )
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
            compute_metrics=compute_metrics,
            weights=torch.tensor(class_weights).float(),
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


def get_model_and_config(
    args,
    model_cls,
    model_config_cls,
    tokenizer,
    dataset,
    nlabels,
    model_load_path,
    pretraining_required,
    scaler=None,
):
    if args.mode == "pre-train" or (
        args.mode == "fine-tune" and (not pretraining_required or args.fromscratch)
    ):
        model_config = model_cls.get_config(
            args=args,
            config_cls=model_config_cls,
            tokenizer=tokenizer,
            dataset=dataset,
            nlabels=nlabels,
        )
        if not args.resume:
            if args.mode == "pre-train":
                logger.info(f"Initializing new {model_cls} model for pre-training.")
            else:
                logger.info(f"Initializing new {model_cls} model for fine-tuning.")
        else:
            logger.info(
                f"Initializing new {model_cls} model for later loading of pre-trained parameters."
            )
        model = model_cls(config=model_config)
        if args.mode == "pre-train":
            model.resize_token_embeddings(len(tokenizer))
    else:
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
        model_config = model_config_cls.from_pretrained(model_load_path)
        model_config.num_labels = int(nlabels)
        model = model_cls.from_pretrained(
            model_load_path,
            config=model_config,
            use_safetensors=False,
        )
        logger.info(
            f"Loaded {model_cls} model with weights from {model_load_path} saved on {datetime.fromtimestamp(model_load_path.stat().st_ctime)} with {n_epochs} epochs trained."
        )
    if args.mode != "pre-train":
        if scaler is not None:
            model.scaler = scaler
        else:
            with open(Path(model_load_path) / "scaler.pkl", "rb") as scaler_file:
                model.scaler = pickle.load(scaler_file)
    return model
