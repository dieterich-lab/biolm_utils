import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _flatten_config(config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary, selectively skipping sections
    based on the execution mode to handle mode-specific duplicate keys.
    """
    flat_config = {}

    # Define which sections to skip for each mode
    skip_sections = {
        "pre-train": ["fine-tuning data source", "inference data source"],
        "tokenize": ["fine-tuning data source", "inference data source"],
        "fine-tune": [
            "tokenizing and pre-training data source",
            "inference data source",
        ],
        "predict": [
            "tokenizing and pre-training data source",
            "fine-tuning data source",
        ],
    }
    sections_to_skip = skip_sections.get(mode, [])

    def _flatten_recursive(sub_dict: Dict[str, Any]):
        for key, value in sub_dict.items():
            if key in sections_to_skip:
                continue  # Skip this entire section

            if isinstance(value, dict):
                _flatten_recursive(value)
            else:
                # No longer checking for duplicates, as we are selectively skipping them.
                flat_config[key] = value

    _flatten_recursive(config)
    return flat_config


def _validate_task_requirement(args: argparse.Namespace):
    """Check that --task is provided for relevant modes."""
    if args.mode in ["fine-tune", "predict", "interpret"] and not args.task:
        raise argparse.ArgumentError(
            None, f"Argument '--task' is required when mode is '{args.mode}'."
        )


def _validate_splitratio(args: argparse.Namespace):
    """Check that --splitratio is a list of 2 or 3 integers summing to 100."""
    if not args.splitratio:
        return
    if not (
        isinstance(args.splitratio, list)
        and all(isinstance(item, int) for item in args.splitratio)
    ):
        raise argparse.ArgumentTypeError("'splitratio' must be a list of integers.")

    if len(args.splitratio) not in [2, 3]:
        raise argparse.ArgumentTypeError(
            f"'splitratio' must contain 2 (train, val) or 3 (train, val, test) values, but got {len(args.splitratio)}."
        )

    if sum(args.splitratio) != 100:
        raise argparse.ArgumentTypeError(
            f"Values in 'splitratio' must sum to 100, but got {sum(args.splitratio)} for {args.splitratio}."
        )


def _validate_split_definitions(args: argparse.Namespace):
    """Check that dev/test splits have the correct structure based on cross-validation."""

    def is_list_of_ints(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, int) for item in value)

    def is_list_of_lists_of_ints(value: Any) -> bool:
        return (
            isinstance(value, list)
            and all(isinstance(sublist, list) for sublist in value)
            and all(isinstance(item, int) for sublist in value for item in sublist)
        )

    for split_name in ["devsplits", "testsplits"]:
        split_value = getattr(args, split_name)
        if not split_value:
            continue

        if args.crossvalidation:
            if not is_list_of_lists_of_ints(split_value):
                raise argparse.ArgumentTypeError(
                    f"With 'crossvalidation', '{split_name}' must be a list of lists of integers (e.g., '[[1, 2], [3]]')."
                )
        else:
            if not is_list_of_ints(split_value):
                raise argparse.ArgumentTypeError(
                    f"Without 'crossvalidation', '{split_name}' must be a flat list of integers (e.g., '[1, 2]')."
                )

    if args.testsplits:
        if not len(args.devsplits) == len(args.testsplits):
            raise argparse.ArgumentTypeError(
                f"With 'testsplits', 'devsplits' and 'testsplits' must have the same length."
            )


def _validate_split_exclusivity(args: argparse.Namespace):
    """
    Check that exactly one of --splitratio or --splitpos is provided.
    """
    ratio_is_set = args.splitratio is not None
    pos_is_set = args.splitpos is not None

    # Condition 1: Not both can be set.
    if ratio_is_set and pos_is_set:
        raise argparse.ArgumentError(
            None,
            f"In mode '{args.mode}', '--splitratio' and '--splitpos' are mutually exclusive. Please provide only one.",
        )

    # Condition 2: At least one must be set.
    if args.mode not in ["tokenize", "predict", "interpret"] and not ratio_is_set and not pos_is_set:
        raise argparse.ArgumentError(
            None,
            f"Either '--splitratio' or '--splitpos' must be provided to define data splits.",
        )


def _validate_splitpos_dependencies(args: argparse.Namespace):
    """Check that if --splitpos is given, --devsplits is also provided."""
    if args.splitpos is not None and args.devsplits is None:
        raise argparse.ArgumentError(
            None,
            "Argument '--devsplits' is required when '--splitpos' is provided.",
        )


def _validate_batchsize(args: argparse.Namespace):
    """Check that if --splitpos is given, --devsplits is also provided."""
    if args.dev and args.batchsize > args.dev:
        args.batchsize = args.dev
        print(f"Setting batchsize to {args.dev}.")


def _validate_all_args(args: argparse.Namespace):
    """Runs all validation checks on the parsed arguments."""
    _validate_task_requirement(args)
    _validate_splitratio(args)
    _validate_split_definitions(args)
    _validate_split_exclusivity(args)
    _validate_splitpos_dependencies(args)
    _validate_batchsize(args)


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with all project arguments."""
    parser = argparse.ArgumentParser(
        description="Run bioinformatical language models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core arguments ---
    core = parser.add_argument_group("Core Arguments")
    core.add_argument(
        "--configfile",
        type=Path,
        help="Path to a YAML config file. Command-line arguments will override config file values.",
    )
    core.add_argument(
        "mode",
        choices=["tokenize", "pre-train", "fine-tune", "predict", "interpret"],
        help="The mode to run the script in.",
    )
    core.add_argument(
        "--outputpath",
        type=Path,
        help="Path to the output directory. If not set, it's derived from the data file name.",
    )
    core.add_argument(
        "--filepath",
        type=Path,
        help="Path to the data file.",
    )
    core.add_argument(
        "--task",
        choices=["regression", "classification"],
        help="The fine-tuning task. Required for modes: fine-tune, predict, interpret.",
    )

    # --- Data Source Arguments ---
    source = parser.add_argument_group("Data Source Arguments")
    source.add_argument(
        "--stripheader",
        action="store_true",
        help="Strip the header from the data file.",
    )
    source.add_argument(
        "--columnsep", type=str, default="\t", help="Column separator in the data file."
    )
    source.add_argument(
        "--tokensep", type=str, help="Separator for tokens within a sequence."
    )
    source.add_argument(
        "--specifiersep",
        type=str,
        help="Separator for token specifiers (e.g., 'A#1.0').",
    )
    source.add_argument(
        "--idpos", type=int, help="Column index (1-based) of the sequence identifier."
    )
    source.add_argument(
        "--seqpos", type=int, help="Column index (1-based) of the sequence."
    )
    source.add_argument(
        "--labelpos", type=int, help="Column index (1-based) of the label."
    )
    source.add_argument(
        "--weightpos",
        type=int,
        default=None,
        help="Column index (1-based) of the regression weights.",
    )
    source.add_argument(
        "--nomarkers",
        action="store_true",
        help="Option to remove `CDS_end` and `-EJ-` markers from the input sequence.",
    )
    parser.add_argument(
        "--_3utr",
        action="store_true",
        help="Trains only on the subsequence after `-CDS_end-`-token.",
    )
    source.add_argument(
        "--non3utr",
        action="store_true",
        help="Trains only on the subsequence until the `CDS_end`-token.",
    )
    source.add_argument(
        "--only512",
        action="store_true",
        help="Filters out sequences that are longer than 512 tokens.",
    )

    # --- Splitting Arguments ---
    splitting = parser.add_argument_group("Splitting Arguments")
    splitting.add_argument(
        "--splitratio",
        type=json.loads,
        help="JSON list for train/val(/test) split ratios, e.g., '[80, 10, 10]'.",
    )
    splitting.add_argument(
        "--crossvalidation",
        type=int,
        default=0,
        help="Number of cross-validation folds. 0 means no CV. 1 is interpreted `True` (for CV on dedicated splits).",
    )
    splitting.add_argument(
        "--splitpos",
        type=int,
        help="Column index (1-based) containing the split identifier.",
    )
    splitting.add_argument(
        "--devsplits",
        type=json.loads,
        help="JSON list of split IDs for the dev set, e.g., '[1, 2]' or '[[1],[2]]' for CV. Required if --splitpos is set.",
    )
    splitting.add_argument(
        "--testsplits",
        type=json.loads,
        help="JSON list of split IDs for the test set, e.g., '[3, 4]' or '[[3],[4]]' for CV. Optional if --splitpos is set.",
    )
    splitting.add_argument(
        "--inferenceonsplits",
        type=json.loads,
        help="JSON list of split IDs to run inference on, e.g., '[1, 2]'.",
    )

    # --- Tokenization Arguments ---
    tokenization = parser.add_argument_group("Tokenization Arguments")
    tokenization.add_argument(
        "--encoding",
        choices=["bpe", "3mer", "5mer", "atomic"],
        help="Tokenization encoding method.",
    )
    tokenization.add_argument(
        "--samplesize",
        type=int,
        default=None,
        help="Downsample the data to this size before training tokenizer.",
    )
    tokenization.add_argument(
        "--vocabsize", type=int, default=20000, help="Vocabulary size for BPE."
    )
    tokenization.add_argument(
        "--minfreq",
        type=int,
        default=2,
        help="Minimum frequency for a token to be in the BPE vocabulary.",
    )
    tokenization.add_argument(
        "--maxtokenlength",
        type=int,
        default=10,
        help="Maximum length of a token in the BPE vocabulary.",
    )
    tokenization.add_argument(
        "--atomicreplacements",
        type=json.loads,
        help='JSON dict for replacing tokens, e.g., \'{"-CDSstop": "s"}\'.',
    )
    tokenization.add_argument(
        "--centertoken",
        type=str,
        help="If an input string exceeds max length, it is centered around this token.",
    )
    tokenization.add_argument(
        "--lefttailing",
        action="store_true",
        help="Truncation is performed from the left side of the tokenized input sequence.",
    )

    # --- Training arguments ---
    training = parser.add_argument_group("Training Arguments")
    training.add_argument(
        "--nepochs", type=int, default=2, help="Number of training epochs."
    )
    training.add_argument(
        "--batchsize", type=int, default=16, help="Batch size for training."
    )
    training.add_argument(
        "--gradacc", type=int, default=1, help="Gradient accumulation steps."
    )
    training.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping."
    )
    training.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="Maximal input sequence length for the model.",
    )
    training.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    training.add_argument(
        "--fromscratch",
        action="store_true",
        help="Fine-tune a model with freshly initialized weights. No function if model doesn't need pre-training.",
    )
    training.add_argument(
        "--scaling",
        choices=["log", "minmax", "standard"],
        default="log",
        help="Method for scaling regression labels.",
    )
    training.add_argument(
        "--weightedregression",
        action="store_true",
        help="Use quality labels as weights for the loss function.",
    )

    # --- Model arguments ---
    model = parser.add_argument_group("Model Arguments")
    model.add_argument(
        "--pretrainedmodel",
        type=Path,
        default=None,
        help="Path to a pretrained model or experiment directory to continue from.",
    )
    model.add_argument(
        "--model_config", type=str, help="Name of the model configuration to use."
    )

    # --- Interpretation Arguments ---
    interp = parser.add_argument_group("Interpretation Arguments")
    interp.add_argument(
        "--handletokens",
        choices=["remove", "mask", "replace"],
        help="Method for LOO score calculation.",
    )
    interp.add_argument(
        "--replacementdict",
        type=json.loads,
        help="JSON dict of tokens to replace for LOO scores.",
    )
    interp.add_argument(
        "--replacespecifier",
        action="store_true",
        help="If True, also replace specifiers during LOO.",
    )

    # --- Debugging and Environment Arguments ---
    debug = parser.add_argument_group("Debugging and Environment Arguments")
    debug.add_argument(
        "--dev",
        default=False,
        nargs="?",
        const=16,
        type=lambda x: False if x.lower() == "false" else int(x),
        help="Run in development mode (e.g., smaller dataset). Use 'False' to disable, or provide an integer for dataset size.",
    )
    debug.add_argument(
        "--silent", action="store_true", help="Silence non-essential logging."
    )
    debug.add_argument(
        "--accelerator",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Hardware accelerator to use.",
    )
    debug.add_argument(
        "--getdata",
        action="store_true",
        help="Only tokenize and save the data to file, then quit.",
    )
    debug.add_argument(
        "--forcenewdata",
        action="store_true",
        help="Forces creation of a dataset, even if a dataset file already exists.",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses arguments, handling config file loading, overrides, and validation.
    """
    parser = create_parser()

    # --- First pass: get config file path and mode ---
    # We need the mode to correctly flatten the config file.
    temp_args, _ = parser.parse_known_args(args)

    config_defaults: Dict[str, Any] = {}
    if temp_args.configfile:
        if temp_args.configfile.exists():
            with open(temp_args.configfile, "r") as f:
                nested_config = yaml.safe_load(f)
                # Flatten the config, selectively skipping sections based on mode
                config_defaults = _flatten_config(nested_config, temp_args.mode)
                # Handle boolean 'crossvalidation' from YAML before setting defaults
                if "crossvalidation" in config_defaults and isinstance(
                    config_defaults["crossvalidation"], bool
                ):
                    config_defaults["crossvalidation"] = (
                        1 if config_defaults["crossvalidation"] else 0
                    )
        else:
            raise FileNotFoundError(f"Config file not found: {temp_args.configfile}")

    # --- Set defaults from config and then parse all arguments ---
    parser.set_defaults(**config_defaults)
    final_args = parser.parse_args(args)

    # --- Post-parsing validation for argument dependencies ---
    _validate_all_args(final_args)

    return final_args


if __name__ == "__main__":
    try:
        args = parse_args()
        print("Final arguments:")
        print(args)
    except (argparse.ArgumentError, argparse.ArgumentTypeError, ValueError) as err:
        print(f"Configuration Error: {err}")
