import logging
import os
from datetime import datetime
from pathlib import Path

from transformers import Trainer

from biolm_utils.params import parse_args
from biolm_utils.rna_datasets import RNACNNDataset, RNALanguageDataset
from biolm_utils.train_utils import (compute_metrics_for_classification,
                                     compute_metrics_for_regression)
from biolm_utils.trainer import (RegressionTrainer, WeightedRegressionTrainer,
                                 WeightedSamplingTrainer)

# Get the arguments from the command line.
args = parse_args()

# Switch off the 'The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.' warning if desired.
if args.silent:
    logging.getLogger("accelerate").setLevel(logging.WARNING)

if args.outputpath is None:
    args.outputpath = Path(args.filepath).stem

OUTPUTPATH = Path(args.outputpath)
OUTPUTPATH.mkdir(parents=True, exist_ok=True)

MODELLOADPATH = None
TOKENIZERFILE = OUTPUTPATH / "tokenizer.json"
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
    format=f"%(asctime)s ({args.mode} {OUTPUTPATH.stem}) - %(message)s",
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

REGRESSIONTRAINER_CLS = (
    WeightedRegressionTrainer if args.weightedregression else RegressionTrainer
)

# CLASSIFICATIONTRAINER_CLS = Trainer
CLASSIFICATIONTRAINER_CLS = WeightedSamplingTrainer

DATASET_CLS = RNALanguageDataset if args.encoding == "bpe" else RNACNNDataset

MLMTRAINER_CLS = Trainer

METRIC = (
    compute_metrics_for_classification
    if args.task == "classification"
    else compute_metrics_for_regression
)
