"""Cross-validation and orchestration helpers.

This module provides the CrossValidator class: an explicit, testable,
easy-to-read replacement for the older decorator-based `parametrized_decorator`.

For backward compatibility we still provide `parametrized_decorator` as a thin
wrapper that instantiates CrossValidator and runs it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from .dataset_utils import (
    check_batchsize,
    log_classification_counts,
    make_subsets,
    split_indices,
)
from .paths import Paths

logger = logging.getLogger(__name__)


def _log_split_summary(results: List[Any], res_type: str) -> None:
    """Log mean and std of the per-split metrics.

    run_once_fn returns one ``{"metric": value}`` dict per split (see
    runner._train/_test); plain numbers are accepted as well.
    """
    values = []
    for res in results:
        value = res.get("metric") if isinstance(res, dict) else res
        if isinstance(value, (int, float, np.number)):
            values.append(float(value))
    if not values:
        logger.info(f"No numeric {res_type} results to summarize: {results}")
        return
    logger.info(
        f"Mean {res_type} results from {len(values)} splits: {np.mean(values)}, "
        f"Std: {np.std(values)} (per split: {values})"
    )


class CrossValidator:
    """Orchestrates runs over train/val/test splits and modes.

    Usage:
        cv = CrossValidator(params, dataset, run_once_fn, base_paths)
        results = cv.execute()

    run_once_fn must accept signature:
      (train_dataset, val_dataset, test_dataset, model_load_path, model_save_path, report_file, rank_file)

    The class intentionally does not mutate module globals; instead it passes
    Paths objects to the underlying run function.
    """

    def __init__(
        self,
        params: Any,
        dataset: Any,
        run_once_fn: Callable,
        base_paths: Paths,
    ):
        self.params = params
        self.dataset = dataset
        self.run_once_fn = run_once_fn
        self.base_paths = base_paths
        # Validate params for consistent cross-validation behavior early so
        # we surface clear errors instead of obscure failures downstream.
        self._validate_params()

    def execute(self):
        m = self.params.mode

        # Tokenize handled earlier by biolm.py (short-circuit), but keep guard
        if m == "tokenize":
            return self.run_once_fn(None, None, None, None, None, None, None)

        if m == "pre-train":
            return self._run_pretrain()

        if m == "fine-tune":
            ds = getattr(self.params, "data_source", None)
            cv = (
                getattr(ds, "crossvalidation", None)
                if ds is not None
                else getattr(self.params, "crossvalidation", None)
            )
            if cv:
                # predefined splits use splitpos; otherwise random crossval
                splitpos = (
                    getattr(ds, "splitpos", None)
                    if ds is not None
                    else getattr(self.params, "splitpos", None)
                )
                if splitpos:
                    return self._run_predefined_splits()
                else:
                    return self._run_random_crossval()
            else:
                # no cross-validation
                ds = getattr(self.params, "data_source", None)
                splitpos = (
                    getattr(ds, "splitpos", None)
                    if ds is not None
                    else getattr(self.params, "splitpos", None)
                )
                devsplits = (
                    getattr(ds, "devsplits", None)
                    if ds is not None
                    else getattr(self.params, "devsplits", None)
                )
                if splitpos and devsplits:
                    return self._run_predefined_no_cv()
                elif (
                    getattr(ds, "splitratio", None)
                    if ds is not None
                    else getattr(self.params, "splitratio", None)
                ):
                    return self._run_single_random_split()
                else:
                    # fallback to a single full-dataset training
                    return self._run_pretrain()

        if m in ("predict", "interpret"):
            return self._run_predict()

        raise ValueError(f"Unsupported mode: {m}")

    def _validate_params(self) -> None:
        """Validate crossvalidation-related parameter combinations and raise
        helpful errors for invalid or ambiguous configurations.

        Rules enforced:
        - For fine-tune mode:
          * crossvalidation == True requires `splitpos` and `devsplits` (predefined
            splits driven from the dataset file).
          * crossvalidation as integer must be >=2 (k-fold random CV) and requires
            `splitratio` (to generate train/val/test indices).
          * crossvalidation as integer conflicts with `splitpos`: pick either
            predefined splits (bool True) or numeric random CV (int >=2).
        - Without crossvalidation: if `splitpos` is provided, `devsplits` must be
          set (to define validation splits) — otherwise training configuration is
          ambiguous.
        """
        if getattr(self.params, "mode", None) != "fine-tune":
            return

        # Prefer nested data_source values when available
        ds = getattr(self.params, "data_source", None)
        cv = (
            getattr(ds, "crossvalidation", None)
            if ds is not None
            else getattr(self.params, "crossvalidation", None)
        )
        splitpos = (
            getattr(ds, "splitpos", None)
            if ds is not None
            else getattr(self.params, "splitpos", None)
        )
        splitratio = (
            getattr(ds, "splitratio", None)
            if ds is not None
            else getattr(self.params, "splitratio", None)
        )
        devsplits = (
            getattr(ds, "devsplits", None)
            if ds is not None
            else getattr(self.params, "devsplits", None)
        )

        # Normalized checks
        if cv:
            # Boolean True -> predefined splits required
            if isinstance(cv, bool) and cv is True:
                if not splitpos:
                    raise ValueError(
                        "crossvalidation=True requires 'splitpos' to be set so predefined splits can be used. "
                        "If you want random k-fold CV, set crossvalidation to an integer >= 2 and set 'splitratio'."
                    )
                if not devsplits:
                    raise ValueError(
                        "crossvalidation=True with 'splitpos' requires 'devsplits' to be defined in your config (testsplits is optional)."
                    )

            # Numeric CV -> require integer >= 2 and splitratio, and forbid splitpos
            elif isinstance(cv, int):
                if cv < 2:
                    raise ValueError(
                        "Numeric crossvalidation must be >= 2 for k-fold cross-validation."
                    )
                if splitpos is not None:
                    raise ValueError(
                        "Numeric crossvalidation (k-fold) conflicts with 'splitpos'. "
                        "Use boolean crossvalidation=True together with 'splitpos' to run CV on predefined splits, or omit splitpos for random k-fold CV."
                    )
                if not splitratio:
                    raise ValueError(
                        "Numeric crossvalidation requires 'splitratio' to be provided (e.g. [80,10,10])."
                    )
            else:
                # Other truthy values are not supported (e.g. strings) -> error early
                raise ValueError(
                    "'crossvalidation' must be either boolean or an integer >= 2 when truthy."
                )

        else:
            # No CV requested. If splitpos is present, caller must provide devsplits.
            if splitpos and not devsplits:
                raise ValueError(
                    "'splitpos' is set but 'devsplits' is missing. Provide 'devsplits' (and optionally 'testsplits') when using 'splitpos' to define deterministic splits."
                )

    # Implementation details mirror the previous behavior but are easier to test.
    def _run_pretrain(self):
        idx = np.arange(len(self.dataset))
        np.random.shuffle(idx)
        dev_flag = getattr(getattr(self.params, "debugging", None), "dev", False)
        train_dataset = make_subsets(
            self.dataset, idx.tolist(), idx.tolist(), None, dev_flag
        )[0]
        val_dataset = None if not dev_flag else train_dataset
        test_dataset = None if not dev_flag else train_dataset
        logger.info(f"Full dataset size: {len(train_dataset)}")
        p = self.base_paths
        return self.run_once_fn(
            train_dataset,
            val_dataset,
            test_dataset,
            p.model_load_path,
            p.model_save_path,
            p.report_file,
            p.rank_file,
        )

    def _run_random_crossval(self):
        results = []
        idx = np.arange(len(self.dataset))
        # cv is expected to be integer here (validated earlier) — prefer nested
        # data_source.crossvalidation
        ds = getattr(self.params, "data_source", None)
        cv = (
            getattr(ds, "crossvalidation", None)
            if ds is not None
            else getattr(self.params, "crossvalidation", None)
        )
        for x in range(int(cv)):
            np.random.shuffle(idx)
            splitratio = (
                getattr(ds, "splitratio", None)
                if ds is not None
                else getattr(self.params, "splitratio", None)
            )
            train_idx, val_idx, test_idx = split_indices(idx, splitratio)
            dev_flag = getattr(self.params, "debugging", None)
            dev_flag = (
                getattr(dev_flag, "dev", False) if dev_flag is not None else False
            )
            train_ds, val_ds, test_ds = make_subsets(
                self.dataset, train_idx, val_idx, test_idx, dev_flag
            )
            batchsize = getattr(
                getattr(self.params, "training", None),
                "batchsize",
                getattr(self.params, "batchsize", None),
            )
            check_batchsize(train_ds, batchsize, "train")
            check_batchsize(val_ds, batchsize, "validation")
            if test_ds is not None:
                check_batchsize(test_ds, batchsize, "test")
            split_paths = self.base_paths.with_split(x, self.params)
            res = self.run_once_fn(
                train_ds,
                val_ds,
                test_ds,
                split_paths.model_load_path,
                split_paths.model_save_path,
                split_paths.report_file,
                split_paths.rank_file,
            )
            results.append(res)
        if self.params.mode != "interpret":
            res_type = "validation" if not test_ds else "test"
            _log_split_summary(results, res_type)
        return results

    def _run_predefined_splits(self):
        # construct a map split -> list indices
        split_dict = {}
        col = getattr(getattr(self.params, "data_source", None), "columnsep", ",")
        pos = getattr(getattr(self.params, "data_source", None), "splitpos", None)
        for i, line in enumerate(self.dataset.lines):
            split = int(line.split(col)[pos - 1].strip('"'))
            split_dict.setdefault(split, []).append(i)

        if getattr(self.params, "data_source", None) and getattr(
            self.params.data_source, "testsplits", None
        ):
            splits = list(
                zip(
                    self.params.data_source.devsplits,
                    self.params.data_source.testsplits,
                )
            )
        else:
            splits = self.params.data_source.devsplits

        results = []
        for k, split in enumerate(splits):
            if getattr(getattr(self.params, "data_source", None), "testsplits", None):
                dev_splits, test_splits = split
            else:
                dev_splits = split
                test_splits = None
            logger.info(f"----- SPLIT {k} -----")
            # build indices
            val_idx = [i for s in dev_splits for i in split_dict[s]]
            if test_splits is not None:
                test_idx = [i for s in test_splits for i in split_dict[s]]
                train_splits = (
                    set(split_dict.keys()) - set(dev_splits) - set(test_splits)
                )
            else:
                test_idx = None
                train_splits = set(split_dict.keys()) - set(dev_splits)
            train_idx = [i for s in train_splits for i in split_dict[s]]
            dev_flag = getattr(getattr(self.params, "debugging", None), "dev", False)
            train_ds, val_ds, test_ds = make_subsets(
                self.dataset, train_idx, val_idx, test_idx, dev_flag
            )
            split_paths = self.base_paths.with_split(k, self.params)
            res = self.run_once_fn(
                train_ds,
                val_ds,
                test_ds,
                split_paths.model_load_path,
                split_paths.model_save_path,
                split_paths.report_file,
                split_paths.rank_file,
            )
            results.append(res)
        if self.params.mode != "interpret":
            res_type = "validation" if not test_ds else "test"
            _log_split_summary(results, res_type)
        return results

    def _run_predefined_no_cv(self):
        # deterministic splitpos + devsplits + optional testsplits
        split_dict = {}
        for i, line in enumerate(self.dataset.lines):
            split = int(line.split(col)[pos - 1].strip('"'))
            split_dict.setdefault(split, []).append(i)

        train_splits = set(split_dict.keys()) - set(self.params.data_source.devsplits)
        if getattr(self.params.data_source, "testsplits", None):
            train_splits -= set(self.params.data_source.testsplits)
        train_idx = [i for s in train_splits for i in split_dict[s]]
        val_idx = [i for s in self.params.data_source.devsplits for i in split_dict[s]]
        test_idx = (
            [i for s in self.params.data_source.testsplits for i in split_dict[s]]
            if getattr(self.params.data_source, "testsplits", None)
            else None
        )
        train_ds, val_ds, test_ds = make_subsets(
            self.dataset, train_idx, val_idx, test_idx, self.params.debugging.dev
        )
        logger.info(f"Full dataset size: {len(self.dataset)}")
        logger.info(f"Training dataset size: {len(train_ds)}")
        logger.info(f"Validation dataset size: {len(val_ds)}")
        if test_ds is not None:
            logger.info(f"Test dataset size: {len(test_ds)}")
        p = self.base_paths
        return self.run_once_fn(
            train_ds,
            val_ds,
            test_ds,
            p.model_load_path,
            p.model_save_path,
            p.report_file,
            p.rank_file,
        )

    def _run_single_random_split(self):
        idx = np.arange(len(self.dataset))
        np.random.shuffle(idx)
        splitratio = (
            self.params.data_source.splitratio
            if self.params.data_source.splitratio
            else [80, 20]
        )
        train_idx, val_idx, test_idx = split_indices(idx, splitratio)
        train_ds, val_ds, test_ds = make_subsets(
            self.dataset, train_idx, val_idx, test_idx, self.params.debugging.dev
        )
        logger.info(f"Full dataset size: {len(self.dataset)}")
        logger.info(f"Training dataset size: {len(train_ds)}")
        logger.info(f"Validation dataset size: {len(val_ds)}")
        if test_ds is not None:
            logger.info(f"Test dataset size: {len(test_ds)}")
        check_batchsize(train_ds, self.params.training.batchsize, "train")
        check_batchsize(val_ds, self.params.training.batchsize, "validation")
        if test_ds is not None:
            check_batchsize(test_ds, self.params.training.batchsize, "test")
        p = self.base_paths
        return self.run_once_fn(
            train_ds,
            val_ds,
            test_ds,
            p.model_load_path,
            p.model_save_path,
            p.report_file,
            p.rank_file,
        )

    def _run_predict(self):
        ds = getattr(self.params, "data_source", None)
        inferenceonsplits = (
            getattr(ds, "inferenceonsplits", None)
            if ds is not None
            else getattr(self.params, "inferenceonsplits", None)
        )
        if not inferenceonsplits:
            idx = np.arange(len(self.dataset))
        else:
            idx = [
                i
                for i, line in enumerate(self.dataset.lines)
                if int(
                    line.split(getattr(ds, "columnsep", ","))[
                        getattr(ds, "splitpos", 1) - 1
                    ].strip('"')
                )
                in inferenceonsplits
            ]
        dev_flag = getattr(getattr(self.params, "debugging", None), "dev", False)
        test_dataset = make_subsets(self.dataset, idx, idx, None, dev_flag)[0]
        logger.info(f"Inference dataset size: {len(test_dataset)}")
        p = self.base_paths
        return self.run_once_fn(
            None,
            None,
            test_dataset,
            p.model_load_path,
            p.model_save_path,
            p.report_file,
            p.rank_file,
        )


# Deprecated: parametrized_decorator
# Use CrossValidator + make_run_fn instead.
def parametrized_decorator(params, dataset):
    """Backward-compatible wrapper: returns a decorator that instantiates
    CrossValidator and uses it to call the provided function.
    """

    def decorator(func):
        import warnings

        warnings.warn(
            "parametrized_decorator is deprecated — use CrossValidator + make_run_fn instead. "
            "This wrapper will continue to work for now but will be removed in a future release.",
            DeprecationWarning,
        )

        def wrapper(*args, **kwargs):
            from .path_setup import PathsManager

            paths = PathsManager.get_paths(config=params)
            base_paths = Paths(
                model_load_path=paths["MODELLOADPATH"],
                model_save_path=paths["MODELSAVEPATH"],
                output_path=paths["MODELSAVEPATH"].parent,
                report_file=paths["REPORTFILE"],
                rank_file=paths["RANKFILE"],
            )
            cv = CrossValidator(params, dataset, func, base_paths)
            return cv.execute()

        return wrapper

    return decorator
