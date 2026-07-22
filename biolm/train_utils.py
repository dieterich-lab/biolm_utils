import hashlib
import json
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)
from .metrics import (
    IdentityScaler,
    LogScaler,
    compute_metrics_for_classification,
    compute_metrics_for_regression,
)
from .structured_config import TokenizationConfig
from .tokenization_loader import get_tokenizer
from .trainer_builder import get_trainer

logger = logging.getLogger(__name__)


def _resolve_source_filepath(args) -> Path:
    data_source = getattr(args, "data_source", None)
    filepath = (
        getattr(data_source, "filepath", None) if data_source is not None else None
    )
    if not filepath:
        filepath = getattr(args, "filepath", None)
    if not filepath:
        raise ValueError("data_source.filepath (or filepath) must be set")
    return Path(filepath)


def _compute_source_hash(args) -> str:
    source_path = _resolve_source_filepath(args)
    hasher = hashlib.sha256()
    with open(source_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_metadata(metadata_file: Path) -> dict[str, str]:
    if not metadata_file.exists():
        return {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Unable to read dataset metadata %s", metadata_file)
        return {}


def _apply_model_overrides(model_config, model_overrides):
    if not model_overrides:
        return

    if hasattr(model_overrides, "items"):
        items = model_overrides.items()
    else:
        items = model_overrides.__dict__.items()

    alias_map = {
        "num_layers": "n_layer",
        "num_heads": "n_head",
        "hidden_size": "d_model",
        "intermediate_size": "d_inner",
    }

    for key, value in items:
        if value is None:
            continue
        targets = []
        if key in alias_map:
            targets.append(alias_map[key])
        targets.append(key)
        for attr in targets:
            try:
                setattr(model_config, attr, value)
            except Exception:
                # Some configs guard attribute setting; fall back to dict-style storage
                try:
                    model_config.__dict__[attr] = value
                except Exception:
                    pass


def get_dataset(args, tokenizer, add_special_tokens, dataset_file, dataset_cls):
    current_blocksize = getattr(getattr(args, "training", None), "blocksize", None)
    source_hash = _compute_source_hash(args)
    metadata_file = dataset_file.with_suffix(".metadata.json")
    
    # Extract current parsing configuration to compare against cache
    ds = getattr(args, "data_source", None)
    current_config = {
        "seqpos": getattr(ds, "seqpos", None),
        "idpos": getattr(ds, "idpos", None),
        "labelpos": getattr(ds, "labelpos", None),
        "columnsep": getattr(ds, "columnsep", None),
    }
    force_recreate = getattr(getattr(args, "debugging", None), "forcenewdata", False)

    if dataset_file.exists() and not force_recreate:
        logger.info(f"Loading dataset from {dataset_file}")
        metadata = _load_metadata(metadata_file)
        cached_hash = metadata.get("source_hash")
        cached_config = metadata.get("config", {})

        config_changed = any(cached_config.get(k) != v for k, v in current_config.items())

        if cached_hash is None:
            logger.info("No dataset metadata hash found; forcing recreation.")
            dataset_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        elif cached_hash != source_hash:
            logger.info(
                "Data source changed (%s vs %s); recreating dataset",
                cached_hash,
                source_hash,
            )
            dataset_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        elif config_changed:
            logger.info("Dataset parsing configuration changed; recreating dataset.")
            dataset_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        else:
            with open(dataset_file, "rb") as f:
                dataset = pickle.load(f)
            logger.info(f"First sample length: {len(dataset[0]['input_ids'])}")
            if (
                current_blocksize is not None
                and len(dataset[0]["input_ids"]) != current_blocksize
            ):
                logger.warning(
                    f"Dataset blocksize mismatch ({len(dataset[0]['input_ids'])} vs {current_blocksize}), recreating dataset"
                )
                dataset_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)
            else:
                tokenizer = dataset.tokenizer
                return dataset

    # Create new dataset
    dataset = dataset_cls(
        tokenizer=tokenizer,
        args=args,
        add_special_tokens=add_special_tokens,
    )
    if not getattr(getattr(args, "debugging", None), "dev", False):
        logger.info(f"Saving dataset to {dataset_file}")
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
        metadata = {
            "scaling_method": dataset.scaling_method,
            "source_hash": source_hash,
            "config": current_config,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
    if getattr(getattr(args, "debugging", None), "getdata", False):
        sys.exit()
    return dataset


def create_reports(test_dataset, test_results, scaler, report_path, rank_path):
    seqs = [test_dataset.dataset.seq_idx[i] for i in test_dataset.indices]

    # Get the results.
    preds = test_results.predictions.squeeze()

    # Transform predictions and gold labels to original space.
    preds = scaler.inverse_transform(np.array(preds).reshape(1, -1)).squeeze()

    # Save the sequence idx, predictions and true labels to a csv file.
    if hasattr(test_dataset.dataset, "labels"):
        labels = np.array([test_dataset.dataset.labels[i] for i in test_dataset.indices])
        labels = scaler.inverse_transform(labels.reshape(1, -1)).squeeze()
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
        args.mode == "fine-tune"
        and (
            not pretraining_required
            or getattr(getattr(args, "training", None), "fromscratch", False)
        )
    ):
        model_config = model_cls.get_config(
            args=args,
            config_cls=model_config_cls,
            tokenizer=tokenizer,
            dataset=dataset,
            nlabels=nlabels,
        )
        _apply_model_overrides(model_config, getattr(args, "model", None))
        if not getattr(getattr(args, "training", None), "resume", False):
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
        )
        logger.info(
            f"Loaded {model_cls} model with weights from {model_load_path} saved on "
            f"{datetime.fromtimestamp(model_load_path.stat().st_ctime)} with {n_epochs} epochs trained."
        )
        model.scaling_method = getattr(model.config, "scaling_method", None)
    if scaler is not None:
        model.scaler = scaler
    elif not hasattr(model, "scaler"):
        model.scaler = IdentityScaler()
    return model
