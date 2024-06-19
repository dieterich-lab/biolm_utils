import logging
from collections import defaultdict

import numpy as np
from torch.utils.data import Subset

from entry import MODELLOADPATH, MODELSAVEPATH, OUTPUTPATH, RANKFILE, REPORTFILE

logger = logging.getLogger(__name__)


def cv_wrapper(args, dataset):
    """
    This is wrapper function that wraps about the `run` function and takes care of cross validation of the data.
    Cross valdiation splits are currently available for RNA/protein data and MLM/regression tasks.
    """
    # Tokenization is the simples of cases and doesn't need cv.
    if args.mode == "tokenize":

        def inner(func):
            return func(None, None, None, None, None)

        return inner

    # For the following data, we actually do cross validation.
    if args.mode not in ["pre-train", "predict"] and args.splitpos is not None:
        # Seperate the data splits into a dictionary.
        split_dict = defaultdict(list)
        for i, line in enumerate(dataset.lines):
            split_dict[int(line.split(args.columnsep)[args.splitpos - 1])].append(i)

        # This is the actual wrapper that iterates over the splits and collect the results.
        def cross_validate(func):
            global MODELLOADPATH, MODELSAVEPATH, REPORTFILE, RANKFILE, OUTPUTPATH

            # This list collects the results of the individual splits.
            results = list()
            for test_split in range(len(split_dict)):

                # We'll change the paths to save model and outputs for each split seperately.
                MODELSAVEPATH = MODELSAVEPATH / f"{test_split}"
                OUTPUTPATH = OUTPUTPATH / f"{test_split}"
                REPORTFILE = REPORTFILE.parent / f"{test_split}" / REPORTFILE.name

                if args.mode == "fine-tune":
                    RANKFILE = RANKFILE.parent / f"{test_split}" / RANKFILE.name
                if args.mode == "interpret":
                    MODELLOADPATH = MODELLOADPATH / f"{test_split}"

                # Define the validation split id.
                val_split = (test_split - 1) % len(split_dict)

                # Get the validation and test idx.
                val_idx = split_dict[val_split]
                test_idx = split_dict[test_split]

                # Define the trianing split idx.
                train_splits = set(range(len(split_dict))) - {test_split, val_split}
                train_idx = list()

                # Collect the training idx.
                for s in train_splits:
                    train_idx += split_dict[s]

                # Create the datasets.
                if not args.dev:
                    test_dataset = Subset(dataset, test_idx)
                    train_dataset = Subset(dataset, train_idx)
                    val_dataset = Subset(dataset, val_idx)
                else:
                    train_dataset = test_dataset = val_dataset = Subset(
                        dataset, np.arange(len(dataset))
                    )

                logger.info(f"Split {test_split}")
                logger.info(
                    f"Len train dataset: {len(train_dataset)}, len val dataset: {len(val_dataset)}, len test dataset: {len(test_dataset)}"
                )

                # Train, test or interpret and collect results.
                res = func(
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    MODELLOADPATH,
                    MODELSAVEPATH,
                )
                results.append(res)

                # Change paths back.
                MODELSAVEPATH = MODELSAVEPATH.parent
                OUTPUTPATH = OUTPUTPATH.parent
                REPORTFILE = REPORTFILE.parent.parent / REPORTFILE.name
                if args.mode == "fine-tune":
                    RANKFILE = RANKFILE.parent.parent / RANKFILE.name
                if args.mode == "interpret":
                    MODELLOADPATH = MODELLOADPATH.parent

            if args.mode != "interpret":
                logger.info(f"Mean results: {np.mean(results)}, Std: {np.std(results)}")
                return results

        return cross_validate
    # For the rest of the data, we don't do cross validation
    elif args.mode == "pre-train":

        # Shuffle the data ids.
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)

        if not args.dev:
            train_dataset = Subset(dataset, idx)
            val_dataset = None
            test_dataset = None
        # These are debugging settings.
        else:
            train_dataset = test_dataset = val_dataset = Subset(dataset, idx)

        def validate(func):
            return func(
                train_dataset, val_dataset, test_dataset, MODELLOADPATH, MODELSAVEPATH
            )

        return validate
    elif args.mode == "fine-tune":

        # Shuffle the data ids.
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)

        train_idx, val_idx = idx[: int(len(idx) * 0.9)], idx[int(len(idx) * 0.9) :]

        if not args.dev:
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            test_dataset = None
        # These are debugging settings.
        else:
            train_dataset = test_dataset = val_dataset = Subset(dataset, idx)

        def validate(func):
            return func(
                train_dataset, val_dataset, test_dataset, MODELLOADPATH, MODELSAVEPATH
            )

        return validate
    elif args.mode == "predict":
        idx = np.arange(len(dataset))
        test_dataset = Subset(dataset, idx)

        def validate(func):
            return func(None, None, test_dataset, MODELLOADPATH, MODELSAVEPATH)

        return validate
