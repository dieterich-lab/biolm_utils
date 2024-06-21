import argparse
import sys
from collections.abc import MutableMapping

import yaml


def none_or_int(x):
    return eval(x)


def geq_one(value):
    value = int(value)
    if value <= 1:
        raise argparse.ArgumentTypeError(
            f"Invalid batch size ({value}). Batch size must be bigger than one due to `batchnorm1d` constraints."
        )
    return value


def parse_args(*args):
    parser = argparse.ArgumentParser(prog="BLM", add_help=False)

    # Following are the main parameters for either training a tokenizer or training a model.
    parser.add_argument(
        "mode",
        choices=["tokenize", "pre-train", "fine-tune", "predict", "interpret"],
        help="""
        `tokenize` = Tokenizing your data and saving the tokenizer.
        `pre-train` = Masked Language Modelling (MLM) on a pre-traning dataset.
        `fine-tune` = Regression fine-tuning for a pre-trained LM or adhoc on a fine-tuning dataset.
        `predict` = Running inference of a fine-tuned model on a dataset.
        `interpret` = Extraction of loo scores using the fine-tuned model on the fine-tuning dataset.
        """,
    )

    # Following identifies the data source.
    parser.add_argument("--filepath", type=str, help="The path the data file.")
    parser.add_argument(
        "--outputpath",
        type=str,
        default=None,
        help="Optional outputpath during the `predict` step, will revert to filepath if `None`.",
    )

    parser.add_argument(
        "--stripheader",
        action="store_true",
        help="If the file has a header, turn on this option to discard it.",
    )
    parser.add_argument(
        "--columnsep",
        type=str,
        default=",",
        help="Separating character for the the different columns in the file",
    )
    parser.add_argument(
        "--tokensep",
        type=str,
        default=None,
        help="Separator for atomic tokens in your sequence.",
    )
    parser.add_argument(
        "--specifiersep",
        type=str,
        default=None,
        help="""
        Atomic encoding only. If inputs are further specified with a float number, this is the separator
        that it should be separated by, e.g. '...,a#2.5,...' if 'a" is further specified by '2.5'.
        """,
    )
    parser.add_argument(
        "--seqpos",
        type=int,
        help="Field position of the sequence in the data file (for 'our' datasets, this will be fixed in `entry.py`).",
    )
    parser.add_argument(
        "--idpos",
        type=int,
        help="Field position of the sequence in the data file (for 'our' datasets, this will be fixed in `entry.py`).",
    )
    parser.add_argument(
        "--splitpos",
        type=none_or_int,
        default=None,
        help="The field position of the split identifier of the split. or 'None' if no cross validation is desired.",
    )
    parser.add_argument(
        "--labelpos",
        type=int,
        help="Field position of the label in the data file  (for 'our' datasets, this will be fixed in `entry.py`).",
    )
    parser.add_argument(
        "--weightpos",
        type=int,
        help="Field position of the regression weights in the data file.",
    )
    parser.add_argument(
        "--pretrainedmodel",
        default=None,
        help="""
        If not set, the tokenizer/pre-trained model will be inferred from the outputpath.
        When pre-traning MLM this refers to using the tokenizer of differenly named run.
        When fine-tuning, this refers to using a pre-trained model from a differenly named run.
        Otherwise the pretrainedmodel is derived according to the `filepath`/`outputpath`.
        """,
    )
    parser.add_argument(
        "--encoding",
        type=str,
        nargs="?",
        default="bpe",
        choices=["bpe", "3mer", "5mer", "atomic"],
        help="""
        Defines how to tokenize an input string. 
        `bpe` is Byte Pair Encoding.
        `3mer` and `5mer` correspont to non overlapping 3-/5-grams.
        `atomic equals character-level tokenization.
        """,
    )

    # If you want to tokenize, you only need to specify the following.
    parser.add_argument(
        "--samplesize",
        type=int,
        default=None,
        help="If your sample data is to big, you can downsample it",
    )
    parser.add_argument(
        "--vocabsize",
        type=int,
        default=20_000,
        help="Determines the final vocabulary size while during byte pair encoding",
    )

    parser.add_argument(
        "--minfreq",
        type=int,
        default=2,
        help="Determines the minimal frequency of a token to be included in the BPE vocabulary.",
    )
    parser.add_argument(
        "--maxtokenlength",
        type=int,
        default=10,
        help="Determines how long a token may be at max in the final BPE vocab.",
    )
    parser.add_argument(
        "--atomicreplacements",
        type=str,
        default=None,
        help="A dictionary-like string that contains the replacements of multi character tokens to atomic characters of the BPE-alphabet, i.e. `{'-CDSstop': 's'}`.",
    )
    parser.add_argument(
        "--centertoken",
        type=str,
        help="If the input string extends the 512 token length, it is centered around the given token.",
    )
    parser.add_argument(
        "--nomarkers",
        action="store_true",
        help="Option to remove `CDS_end`/`-CDSstop-` and `-EJ-` from the input sequence.",
    )
    parser.add_argument(
        "--_3utr",
        action="store_true",
        help="Trains only on the subsequence after `-CDS_end-`-token.",
    )
    parser.add_argument(
        "--non3utr",
        action="store_true",
        help="Trains only on the subsequence until `CDS_end`-token.",
    )
    parser.add_argument(
        "--only512",
        action="store_true",
        help="Filters out sequences that > 512 tokens.",
    )
    parser.add_argument(
        "--lefttailing",
        action="store_true",
        help="Truncation is done from the left side of the tokenized input sequence.",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Number of GPUs that is being trained on (only even numbers allowed).",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="Option to train on GPU or CPU.",
    )
    parser.add_argument(
        "--batchsize",
        default=16,
        type=geq_one,
        help="""
        This batch size will be multiplied by 4 with gradient accumulation. If you don't want this, change `gradacc` to the desired value.
        Also, we prohibit batch sizes <2 and advise the user to batch sizes >8 as batch normalization will suffer elsewise.
        """,
    )
    parser.add_argument(
        "--learningrate",
        type=float,
        help="Denote a specific learning rate",
    )
    parser.add_argument(
        "--gradacc",
        default=4,
        type=int,
        help="""The number of batches to be aggregated before calculating gradients.
        With a `batchsize` of 16, the effective batch size will 64.
        Default is set to `4` and shoould not be lowered as we account for GPU parallelization with it.
        This guarantees that we will always have the same effective batch size.""",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=512,
        help="Maximal input sequence length of the model.",
    )
    parser.add_argument("--nepochs", default=50, type=int)
    parser.add_argument(
        "--patience",
        nargs="?",
        const=2,
        default=2,
        type=int,
        help="Number of epochs without improvement on the development set before training stops.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        default=False,
        const=True,
        type=int,
        help="""
        This parameter is overloaded with two options:
        1) `--resume` (without parameters) triggers the huggingface internal `resume_from_checkpoint` option which will only _continue_
        a training that has been interrupted. For example, a planned training that was to run for 50 epochs and was interrupted  at epoch
        23 can be resumed from the best checkpoint to be run from epoch 23 to planned epoch 50.
        2) `--resume X` will trigger further pre-training a model from its best checkpoint for additional `X` epochs.
        """,
    )
    parser.add_argument(
        "--fromscratch",
        action="store_true",
        help="Finetunes a regression model on a given task with freshly initialized parameters.",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="log",
        choices=["log", "minmax", "stanard"],
    )
    parser.add_argument(
        "--weightedregression",
        action="store_true",
        help="Uses quality labels as weights for the loss function.",
    )
    parser.add_argument(
        "--handletokens",
        type=str,
        choices=["remove", "mask", "replace"],
        help="How to handle 'missing' tokens during interpretability calculations.",
    )

    # Debugging options
    parser.add_argument(
        "--silent",
        action="store_true",
        help="If set to True, verbose printing of the transformers library is disabled. Only results are printed.",
    )
    parser.add_argument(
        "--dev",
        nargs="?",
        const=16,
        default=False,
        type=int,
        help="A flag to speed up processes for debugging by sampling down training data to the given amount of samples and using this data also for validation steps.",
    )
    parser.add_argument(
        "--getdata",
        action="store_true",
        help="Only tokenize and save the data to file, then quit.",
    )
    parser.add_argument(
        "--saveastensors",
        action="store_true",
        help="Some datasets are small enough that we can save them as tensors objects instead of plain tokenized ids.",
    )

    # Or simply pass a config file.
    parser.add_argument(
        "--configfile",
        type=str,
        default=None,
        help="Path to the a config file that will overrule CLI arguments.",
    )
    if args:
        params = parser.parse_args(*args)
    else:
        params = parser.parse_args()

    if params.configfile is not None:
        configfile = params.configfile

        # local function to get the parameters from (nested) yaml sections.
        def flatten_dict_gen(d):
            for k, v in d.items():
                if params.mode in ["pre-train", "tokenize"] and (
                    k == "fine-tuning data source" or k == "inference data source"
                ):
                    continue
                if params.mode == "fine-tune" and (
                    k == "tokenizing and pre-training data source"
                    or k == "inference data source"
                ):
                    continue
                if params.mode == "predict" and (
                    k == "tokenizing and pre-training data source"
                    or k == "fine-tuning data source"
                ):
                    continue
                if isinstance(v, MutableMapping) and k != "atomicreplacements":
                    yield from dict(flatten_dict_gen(v)).items()
                else:
                    yield k, v

        with open(params.configfile, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                raise
        sys.argv = [
            x
            for i, x in enumerate(sys.argv, 1)
            if x != "--configfile" and sys.argv[i - 2] != "--configfile"
        ]
        config = dict(flatten_dict_gen(config))
        optargs = list()
        for kv in config.items():
            if f"--{kv[0]}" in sys.argv:
                continue
            if (isinstance(kv[1], bool) and kv[1] == False) or kv[1] == "None":
                continue
            elif isinstance(kv[1], bool):
                if kv[1]:
                    optargs += [f"--{kv[0]}"]
            else:
                optargs += [f"--{kv[0]}", f"{(kv[1])}"]
        sys.argv += optargs
        params = parser.parse_args()
        params.configfile = configfile
    return params


if __name__ == "__main__":
    params = parse_args()
    print(vars(params))
