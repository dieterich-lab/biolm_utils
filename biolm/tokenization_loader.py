"""Tokenizer loading utilities extracted from train_utils."""

import json
import logging
import tempfile
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def inject_data_source_config_into_tokenizer(tokenizer_json_dict, args):
    """
    Injects dynamic parsing configuration (like column separator, seqpos, and token separator)
    into the JSON configuration of a loaded tokenizer.
    """
    col = getattr(getattr(args, "data_source", None), "columnsep", "\t")
    seqpos_value = getattr(getattr(args, "data_source", None), "seqpos", None)
    if seqpos_value is None:
        seqpos_value = 1
        
    if "pre_tokenizer" in tokenizer_json_dict and "pretokenizers" in tokenizer_json_dict["pre_tokenizer"]:
        if len(tokenizer_json_dict["pre_tokenizer"]["pretokenizers"]) > 2:
            tokenizer_json_dict["pre_tokenizer"]["pretokenizers"][1]["pattern"]["Regex"] = f"^([^{col}]*{col}){{{int(seqpos_value) - 1}}}"
            tokenizer_json_dict["pre_tokenizer"]["pretokenizers"][2]["pattern"]["Regex"] = f"{col}.*"
            
    tokensep = getattr(getattr(args, "data_source", None), "tokensep", None)
    if tokensep is not None and "normalizer" in tokenizer_json_dict and "normalizers" in tokenizer_json_dict["normalizer"]:
        num_elements = len(tokenizer_json_dict["normalizer"]["normalizers"])
        if num_elements > 1:
            tokenizer_json_dict["normalizer"]["normalizers"][-2]["pattern"]["String"] = tokensep
        else:
            encoding = getattr(getattr(args, "tokenization", None), "encoding", "atomic")
            replacement = "" if encoding == "bpe" else " "
            pattern = {"type": "Replace", "pattern": {"String": tokensep}, "content": replacement}
            tokenizer_json_dict["normalizer"]["normalizers"].insert(0, pattern)



def get_tokenizer(args, tokenizer_file: Path, tokenizer_cls, pretraining_required):
    """Load tokenizer for pre-train/fine-tune/predict/interpret modes."""

    mode = getattr(args, "mode", None)

    if mode == "fine-tune" and pretraining_required:
        tokenizer_config_file = (
            tokenizer_file.parent / "pre-train" / "tokenizer_config.json"
        )
        tokenizer_file = tokenizer_file.parent / "pre-train" / "tokenizer.json"
        with open(tokenizer_config_file, "r") as ff:
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
            f"Loaded tokenizer config from {tokenizer_config_file} and setting it to {model_max_len} model max length"
        )
        with open(tokenizer_file, "r") as f:
            tokenizer_json = json.load(f)
            
        inject_data_source_config_into_tokenizer(tokenizer_json, args)
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
    else:
        blocksize = getattr(getattr(args, "training", None), "blocksize", None)
        logger.info(
            f"Loading tokenizer from {tokenizer_file} and setting it to {blocksize} model max length"
        )

        tokenizer_dir = tokenizer_file
        if (
            not tokenizer_dir.exists()
            or not (tokenizer_dir / "tokenizer_config.json").exists()
        ):
            parent_dir = tokenizer_file.parent
            logger.info(
                f"Tokenizer config not found in {tokenizer_dir}; trying {parent_dir}"
            )
            tokenizer_dir = parent_dir
        tokenizer_kwargs = {
            "model_max_length": blocksize,
            "truncation": True,
            "truncation_side": (
                "left"
                if getattr(getattr(args, "tokenization", None), "lefttailing", False)
                else "right"
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            if tokenizer_dir.exists():
                for item in tokenizer_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, tmpdir_path)
            
            tmp_tokenizer_json = tmpdir_path / "tokenizer.json"
            if tmp_tokenizer_json.exists():
                with open(tmp_tokenizer_json, "r") as f:
                    tokenizer_json_dict = json.load(f)
                    
                inject_data_source_config_into_tokenizer(tokenizer_json_dict, args)
                        
                with open(tmp_tokenizer_json, "w") as f:
                    json.dump(tokenizer_json_dict, f)

            try:
                tokenizer = tokenizer_cls.from_pretrained(
                    str(tmpdir_path),
                    **tokenizer_kwargs,
                )
            except (TypeError, OSError, FileNotFoundError) as exc:
                logger.warning(
                    "Tokenizer %s couldn't be loaded via from_pretrained: %s",
                    tokenizer_cls.__name__,
                    exc,
                )
                tokenizer_json = tmpdir_path / "tokenizer.json"
                if not tokenizer_json.exists():
                    raise

                tokenizer_config_json = tmpdir_path / "tokenizer_config.json"
                config_overrides = {}
                if tokenizer_config_json.exists():
                    with open(tokenizer_config_json, "r") as cfg:
                        tok_config = json.load(cfg)
                    for key in [
                        "cls_token",
                        "unk_token",
                        "mask_token",
                        "pad_token",
                        "sep_token",
                        "bos_token",
                        "eos_token",
                        "model_max_length",
                        "truncation_side",
                    ]:
                        if key in tok_config:
                            config_overrides[key] = tok_config[key]

                tokenizer = tokenizer_cls(
                    tokenizer_file=str(tokenizer_json),
                    **{**tokenizer_kwargs, **config_overrides},
                )
                logger.warning(
                    "Loaded tokenizer directly from %s due to %s", tokenizer_json, exc
                )
        tokenizer.name_or_path = tokenizer_file
    return tokenizer
