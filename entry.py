import logging
import os
from datetime import datetime
from pathlib import Path

from transformers import (
    BertConfig,
    Trainer,
    XLNetConfig,
)
from transformers.image_processing_utils import BaseImageProcessor

from params import parse_args
from trainer import RegressionTrainer, WeightedRegressionTrainer

# Get the arguments from the command line.
args = parse_args()

# Switch off the 'The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.' warning if desired.
if args.silent:
    logging.getLogger("accelerate").setLevel(logging.WARNING)

if args.outputpath is None:
    args.outputpath = Path(args.filepath).stem

OUTPUTPATH = Path(args.outputpath)
# EXPERIMENTPATH = Path("experiments") / args.experimentname
OUTPUTPATH.mkdir(parents=True, exist_ok=True)

MODELLOADPATH = None
TOKENIZERFILE = OUTPUTPATH / "tokenizer.json"
if args.mode == "fine-tune" and args.model == "xlnet":
    MODELLOADPATH = OUTPUTPATH / "pre-train"
elif args.mode == "interpret":
    MODELLOADPATH = OUTPUTPATH / "fine-tune"

# `pretrainedmodel` changes either:
# - different tokenizer when pre-training
# - different pre-trained-model/tokenizer when fine-tuning
# - tokenizer/fine-tuned model path for inference
if args.pretrainedmodel:
    if args.mode != "pre-train":
        MODELLOADPATH = Path(args.pretrainedmodel)
        TOKENIZERFILE = MODELLOADPATH / "tokenizer.json"
    else:
        TOKENIZERFILE = Path(args.pretrainedmodel) / "tokenizer.json"

if MODELLOADPATH is not None:
    MODELLOADPATH.mkdir(parents=True, exist_ok=True)

MODELSAVEPATH = OUTPUTPATH / args.mode
if args.mode not in ["tokenize", "predict", "interpret"]:
    MODELSAVEPATH.mkdir(parents=True, exist_ok=True)
REPORTFILE = MODELSAVEPATH / "test_predictions.csv"
RANKFILE = MODELSAVEPATH / "rank_deltas.csv"
TBPATH = MODELSAVEPATH / "tboard"
LOGPATH = MODELSAVEPATH / "logs"
LOGPATH.mkdir(parents=True, exist_ok=True)
TBPATH.mkdir(parents=True, exist_ok=True)

if args.mode != "interpret":
    DATASETFILE = OUTPUTPATH / args.mode / "dataset.json"
else:
    DATASETFILE = OUTPUTPATH / "fine-tune" / "dataset.json"

# Set up logging
now = datetime.now().strftime("%Y-%m-%d_%H:%M")
LOGFILE = LOGPATH / f"{now}.log"
LOGFILE.touch(exist_ok=True)
if not args.dev:
    handlers = [
        logging.FileHandler(LOGFILE, mode="w"),
        logging.StreamHandler(),
    ]
else:
    handlers = [
        logging.StreamHandler(),
    ]
logging.basicConfig(
    format=f"%(asctime)s ({args.mode} {OUTPUTPATH.stem} {args.model}) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=handlers,
)

# We scale the gradient with respect to the number of GPUs to keep an
# effective batch size of `args.batchsize` x `args.gradacc`
if args.dev:
    GRADACC = 1
else:
    GRADACC = args.gradacc / args.ngpus
    logging.info(f"Set gradient accumulation to {GRADACC}.")

# Log the arguments.
logging.info(f"{'=== Params ===':>32}")
for k, v in sorted(vars(args).items()):
    logging.info(f"{k:>25} : {str(v):<25}")


if args.resume == True:
    CHECKPOINTPATH = max(MODELSAVEPATH.glob("checkpoint*"), key=os.path.getmtime)
    logging.info(f"Pretrained model to resume from: {CHECKPOINTPATH}")
else:
    CHECKPOINTPATH = None

REGRESSIONTRAINER = (
    WeightedRegressionTrainer if args.weightedregression else RegressionTrainer
)

MLMTRAINER = Trainer

# Collate models.
MODELDICT = {
    "pre-train": {
    },
    "fine-tune": {
    },
}

# Collate tokenizers.
TOKENIZERDICT = {
}

DATASETDICT = {
}

LEARNINGRATE = 1e-5

MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.0

SPECIAL_TOKENIZER_FOR_TRAINER = None

ADD_SPECIAL_TOKENS = True

CONFIGCLS = XLNetConfig

PRETRAINING_REQUIRED = True