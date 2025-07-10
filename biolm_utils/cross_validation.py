import logging
from collections import Counter, defaultdict
from contextlib import contextmanager

import numpy as np
from torch.utils.data import Subset

from biolm_utils.entry import (
    MODELLOADPATH,
    MODELSAVEPATH,
    OUTPUTPATH,
    RANKFILE,
    REPORTFILE,
)

logger = logging.getLogger(__name__)


def make_datasets(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    dev,
):
    """
    Create train, validation, and test Subset datasets from indices.
    """
    if not dev:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx) if test_idx is not None else None
    else:
        idx = np.arange(len(dataset))
        train_dataset = val_dataset = Subset(dataset, idx.tolist())
        test_dataset = Subset(dataset, idx.tolist()) if test_idx is not None else None
    return train_dataset, val_dataset, test_dataset


def check_batchsize(ds, batchsize, name):
    """
    Raise an exception if the dataset is smaller than the batch size.
    """
    if ds is not None and len(ds) < batchsize:
        raise Exception(
            f"Size of the {name} dataset ({len(ds)}) is smaller than the batch size, please lower the batch size first."
        )


def log_classification_counts(
    params,
    dataset,
    train_dataset,
    val_dataset,
    test_dataset,
):
    """
    Log class distribution for classification tasks.
    """
    if getattr(params, "task", None) == "classification":
        for name, ds in [
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ]:
            if ds is not None:
                counter = Counter(
                    [dataset.LE.classes_[dataset[x]["labels"]] for x in ds.indices]
                )
                logger.info(f"{name} label distribution: {counter}")


def run_and_log(
    func,
    params,
    dataset,
    train_dataset,
    val_dataset,
    test_dataset,
):
    """
    Log dataset info and run the main function.
    """
    log_classification_counts(params, dataset, train_dataset, val_dataset, test_dataset)
    logger.info(
        f"Len train dataset: {len(train_dataset)}, len val dataset: {len(val_dataset)}"
    )
    res = func(
        train_dataset,
        val_dataset,
        test_dataset,
        MODELLOADPATH,
        MODELSAVEPATH,
        REPORTFILE,
        RANKFILE,
    )
    return res


def split_indices(idx, splitratio):
    """
    Split indices into train/val/(test) according to splitratio.
    """
    if len(splitratio) < 3:
        train_idx = idx[: int(len(idx) * splitratio[0] / 100)]
        val_idx = idx[-int(len(idx) * splitratio[1] / 100) :]
        test_idx = None
    else:
        train_end = int(len(idx) * splitratio[0] / 100)
        val_end = train_end + int(len(idx) * splitratio[1] / 100)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
    return train_idx, val_idx, test_idx


@contextmanager
def split_path_context(split_id, params):
    """
    Context manager to handle path changes for each split.
    """
    global MODELSAVEPATH, OUTPUTPATH, REPORTFILE, RANKFILE, MODELLOADPATH
    orig_paths = (
        MODELSAVEPATH,
        OUTPUTPATH,
        REPORTFILE,
        RANKFILE,
        MODELLOADPATH,
    )
    MODELSAVEPATH = MODELSAVEPATH / f"{split_id}"
    OUTPUTPATH = OUTPUTPATH / f"{split_id}"
    REPORTFILE = REPORTFILE.parent / f"{split_id}" / REPORTFILE.name
    if params.mode == "fine-tune":
        RANKFILE = RANKFILE.parent / f"{split_id}" / RANKFILE.name
    if params.mode == "interpret":
        MODELLOADPATH = MODELLOADPATH / f"{split_id}"
    try:
        yield
    finally:
        (
            MODELSAVEPATH,
            OUTPUTPATH,
            REPORTFILE,
            RANKFILE,
            MODELLOADPATH,
        ) = orig_paths


def parametrized_decorator(params, dataset):
    """
    Decorator to wrap the main run function and handle all cross-validation, splitting, and dataset logic.
    """

    def cv_wrapper(func):
        # --- Tokenization mode ---
        if params.mode == "tokenize":

            def tokenize(*args, **kwargs):
                return func(None, None, None, None, None)

            return tokenize

        # --- Cross-validation with splitpos ---
        if params.mode == "fine-tune" and params.crossvalidation and params.splitpos:
            split_dict = dict()
            for i, line in enumerate(dataset.lines):
                split = int(
                    line.split(params.columnsep)[params.splitpos - 1].strip('"')
                )
                if split not in split_dict:
                    split_dict[split] = [i]
                split_dict[split].append(i)

            def cross_validate_on_predefined_splits(*args, **kwargs):
                results = []
                for k, test_split in enumerate(split_dict.keys()):
                    logger.info(f"----- SPLIT {k} -----")
                    with split_path_context(test_split, params):
                        val_pos = (k - 1) % len(split_dict.keys())
                        val_split = list(split_dict.keys())[val_pos]
                        val_idx = split_dict[val_split]
                        test_idx = split_dict[test_split]
                        train_splits = (
                            set(split_dict.keys()) - {val_split} - {test_split}
                        )
                        train_idx = [i for s in train_splits for i in split_dict[s]]

                        train_dataset, val_dataset, test_dataset = make_datasets(
                            dataset, train_idx, val_idx, test_idx, params.dev
                        )
                        res = run_and_log(
                            func,
                            params,
                            dataset,
                            train_dataset,
                            val_dataset,
                            test_dataset,
                            *args,
                        )
                        results.append(res)
                if params.mode != "interpret":
                    res_type = "validation" if not test_dataset else "test"
                    logger.info(
                        f"Mean {res_type} results from {len(results)} splits: {np.mean(results)}, Std: {np.std(results)}"
                    )
                    return results

            return cross_validate_on_predefined_splits

        # --- Cross-validation with splitratio ---
        if (
            params.mode == "fine-tune"
            and params.crossvalidation
            and (params.splitratio or int(params.crossvalidation) > 1)
        ):

            def cross_validate_on_random_splits(*args, **kwargs):
                results = []
                idx = np.arange(len(dataset))
                for x in range(params.crossvalidation):
                    np.random.shuffle(idx)
                    with split_path_context(x, params):
                        train_idx, val_idx, test_idx = split_indices(
                            idx, params.splitratio
                        )
                        train_dataset, val_dataset, test_dataset = make_datasets(
                            dataset, train_idx, val_idx, test_idx, params.dev
                        )
                        check_batchsize(train_dataset, params.batchsize, "train")
                        check_batchsize(val_dataset, params.batchsize, "validation")
                        if test_dataset is not None:
                            check_batchsize(test_dataset, params.batchsize, "test")
                        res = run_and_log(
                            func,
                            params,
                            dataset,
                            train_dataset,
                            val_dataset,
                            test_dataset,
                            *args,
                        )
                        results.append(res)
                if params.mode != "interpret":
                    res_type = "validation" if not test_dataset else "test"
                    logger.info(
                        f"Mean {res_type} results from {len(results)} splits: {np.mean(results)}, Std: {np.std(results)}"
                    )
                    return results

            return cross_validate_on_random_splits

        if params.mode == "fine-tune" and not params.crossvalidation:
            # --- Dedicated splits (no cross-validation, but splitpos/devsplits/testsplits) ---
            if params.splitpos and params.devsplits:

                def run_finetuning_on_predefined_splits(*args, **kwargs):
                    split_dict = defaultdict(list)
                    for i, line in enumerate(dataset.lines):
                        split = int(
                            line.split(params.columnsep)[params.splitpos - 1].strip('"')
                        )
                        split_dict[split].append(i)
                    train_splits = set(set(split_dict.keys())) - set(params.devsplits)
                    if params.testsplits:
                        train_splits -= set(params.testsplits)
                    train_idx = [i for s in train_splits for i in split_dict[s]]
                    val_idx = [i for s in params.devsplits for i in split_dict[s]]
                    if params.testsplits:
                        test_idx = [i for s in params.testsplits for i in split_dict[s]]
                    else:
                        test_idx = None
                    train_dataset, val_dataset, test_dataset = make_datasets(
                        dataset, train_idx, val_idx, test_idx, params.dev
                    )
                    return run_and_log(
                        func,
                        params,
                        dataset,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        *args,
                    )

                return run_finetuning_on_predefined_splits

            # --- Random splits (no cross-validation) ---
            elif params.splitratio:

                def run_finetuning_on_random_splits(*args, **kwargs):
                    idx = np.arange(len(dataset))
                    np.random.shuffle(idx)
                    splitratio = params.splitratio if params.splitratio else [80, 20]
                    train_idx, val_idx, test_idx = split_indices(idx, splitratio)
                    train_dataset, val_dataset, test_dataset = make_datasets(
                        dataset, train_idx, val_idx, test_idx, params.dev
                    )
                    check_batchsize(train_dataset, params.batchsize, "train")
                    check_batchsize(val_dataset, params.batchsize, "validation")
                    if test_dataset is not None:
                        check_batchsize(test_dataset, params.batchsize, "test")
                    return run_and_log(
                        func,
                        params,
                        dataset,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        *args,
                    )

                return run_finetuning_on_random_splits

        # --- Pre-train mode ---
        if params.mode == "pre-train":

            def run_pretraining(*args, **kwargs):
                idx = np.arange(len(dataset))
                np.random.shuffle(idx)
                train_dataset = Subset(dataset, idx)
                val_dataset = test_dataset = (
                    None if not params.dev else Subset(dataset, idx)
                )
                return func(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    MODELLOADPATH,
                    MODELSAVEPATH,
                )

            return run_pretraining

        # --- Predict/Interpret mode ---
        if params.mode in ["predict", "interpret"]:

            def run_prediction(*args, **kwargs):
                if not getattr(params, "inferenceonsplits", None):
                    idx = np.arange(len(dataset))
                else:
                    idx = [
                        i
                        for i, line in enumerate(dataset.lines)
                        if int(
                            line.split(params.columnsep)[params.splitpos - 1].strip('"')
                        )
                        in params.inferenceonsplits
                    ]
                test_dataset = Subset(dataset, idx)
                return func(
                    None,
                    None,
                    test_dataset,
                    MODELLOADPATH,
                    MODELSAVEPATH,
                    REPORTFILE,
                    RANKFILE,
                )

            return run_prediction

    return cv_wrapper
