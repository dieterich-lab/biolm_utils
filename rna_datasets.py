import json
import logging
import re
import tempfile

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import Dataset


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


class RNABaseDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        add_special_tokens,
    ):
        self.tokenizer = tokenizer
        self.args = args

        with open(args.filepath, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
            if args.stripheader:
                lines = lines[1:]

        if args.dev:
            lines = lines[: args.dev // 2] + lines[-args.dev // 2 :]

        # We'll save the original input data lines for later reference.
        self.lines = lines
        self.seq_idx = [
            x.split(args.columnsep)[args.idpos - 1].strip('"') for x in self.lines
        ]

        self.join_str = (
            ""
            if self.args.tokensep is None or args.encoding == "bpe"
            else self.args.tokensep
        )

        # Normalize and pre-trokenize to obtain the sequences.
        normalized_seqs = [
            tokenizer.backend_tokenizer.normalizer.normalize_str(x) for x in lines
        ]

        self.nspecs = 0
        if args.specifiersep is not None:
            with open(tokenizer.name_or_path, "r") as f:
                tokenizer_json = json.load(f)
            tokenizer_json["normalizer"]["normalizers"].pop(-3)
            tokenizer_json["pre_tokenizer"]["pretokenizers"].pop(-1)
            with tempfile.NamedTemporaryFile("r+") as tmp:
                json.dump(tokenizer_json, tmp)
                tmp.seek(0)
                spec_tokenizer = tokenizer.__class__(
                    tokenizer_file=tmp.name,
                    mask_token="[MASK]",
                    cls_token="[CLS]",
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    sep_token="[SEP]",
                    bos_token="[BOS]",
                    eos_token="[EOS]",
                    model_max_length=args.blocksize,
                    truncation=True,
                    truncation_side="left" if args.lefttailing else "right",
                )
            spec_normalized_seqs = [
                spec_tokenizer.backend_tokenizer.normalizer.normalize_str(x)
                for x in lines
            ]
            spec_pre_tokenized_seqs = [
                spec_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)[0][0]
                for x in spec_normalized_seqs
            ]
            self.specs = [
                [
                    re.findall(rf"(?<={args.specifiersep})[^{args.specifiersep}]+", y)
                    for y in x.split(" ")
                ]
                for x in spec_pre_tokenized_seqs
            ]
            self.nspecs = len(max(max([x for x in y]) for y in self.specs))
            self.specs = [
                np.array(
                    [
                        np.pad(
                            list(map(float, y)),
                            (0, self.nspecs - len(y)),
                            constant_values=0.0,
                        )
                        for y in x[: tokenizer.model_max_length]
                    ]
                )
                for x in self.specs
            ]
            self.specs = [
                np.pad(
                    x,
                    ((0, tokenizer.model_max_length - x.shape[0]), (0, 0)),
                    constant_values=0,
                )
                for x in self.specs
            ]

        pre_tokenized_seqs = [
            tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)
            for x in normalized_seqs
        ]
        self.seqs = [
            self.join_str.join([y[0] for y in x]).replace("Ġ", "")
            for x in pre_tokenized_seqs
        ]

        # Set the log level to error to supress the warning that we will
        # actually tokenize sequences which are longer than the model's max sequence length.
        log_lvl = transformers.utils.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        # Evaluate the length of the tokenized unmanipulated/untruncated data.
        if self.args.encoding in ["3mer", "5mer"]:
            self.seqs = self.tokenize_kmers(self.seqs, self.args)
            raw_encodings = self.tokenizer(
                self.seqs,
                add_special_tokens=False,
                truncation=False,
                is_split_into_words=True,
            )["input_ids"]
        else:
            raw_encodings = self.tokenizer(
                self.seqs, add_special_tokens=False, truncation=False
            )["input_ids"]
        # restore log lvl
        transformers.logging.set_verbosity(log_lvl)
        self.tokenized_seqs = [
            self.tokenizer.convert_ids_to_tokens(x) for x in raw_encodings
        ]
        self.tokenized_seqs = [
            list(map(lambda x: x.replace("Ġ", ""), y)) for y in self.tokenized_seqs
        ]
        self.tokenized_seqs = [[x for x in y if x != ""] for y in self.tokenized_seqs]

        # Possible cds-centering.
        if args.centertoken:
            # XXX Check this if that still also holds for k-mers.
            self.seqs = self.get_centered_lines()

        # These two options are actively filtering sequences out and also alter `self.lines`.
        if args.only512:
            self.seqs = self.get_only512()
        if args._3utr:
            self.seqs = self.get_3utr()

        if args.non3utr:
            self.seqs = self.get_non3utr()

        if args.nomarkers:
            self.seqs = self.get_nomarkers()

        encodings = self.tokenizer(
            self.seqs,
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding="max_length",
            is_split_into_words=args.encoding in ["3mer", "5mer"],
        )["input_ids"]

        # Define a one-hot encoder
        self.OHE = self._get_OHE()
        non_special_vocab = [
            v
            for k, v in self.tokenizer.vocab.items()
            if k not in self.tokenizer.special_tokens_map.values()
        ]
        self.OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.OHE.fit([[x] for x in non_special_vocab])

        self.examples = np.array([{"input_ids": e} for e in encodings])

        # TODO: Make this a model attribute
        # Set up the scaler
        if args.scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif args.scaling == "standard":
            self.scaler = StandardScaler()
        elif args.scaling == "log":
            self.scaler = LogScaler()
        else:
            # Not so pretty, but is currently the fastest adaptation for no scaling
            self.scaler = IdentityScaler()

        # get the labels and seq idx for each task.
        if args.mode in ["fine-tune", "predict"] and args.labelpos is not None:
            self.labels = [
                float(x.split(args.columnsep)[args.labelpos - 1].strip('"'))
                for x in self.lines
            ]
            if args.weightpos is not None:
                qualities = [
                    x.split(",")[args.weightpos].strip('"') for x in self.lines
                ]
                qual_dict = {"STRONG": 1.0, "GOOD": 0.75, "WEAK": 0.5, "POOR": 0.25}
                self.qualities = [qual_dict[x] for x in qualities]

            self.labels = self.scaler.fit_transform(
                np.array(self.labels).reshape(-1, 1).astype(float)
            )

            # update self.examples with labels (and quality weights).
            if args.weightpos is None:
                for l, e in zip(self.labels, self.examples):
                    e.update({"labels": l})
            elif args.data == "protein":
                for l, e, q in zip(self.labels, self.examples, self.qualities):
                    e.update({"labels": l})
                    e.update({"qualities": q})

    def __len__(self):
        return len(self.examples)

    def get_centered_lines(self):
        centered_lines = list()
        for line in self.tokenized_seqs:
            if len(line) <= self.tokenizer.model_max_length:
                centered_lines.append(line)
            else:
                cds_pos = [i for i, x in enumerate(line) if self.args.centertoken in x]
                if not cds_pos:
                    centered_lines.append(line)
                else:
                    cds_pos = cds_pos[0]
                    middle = (self.tokenizer.model_max_length - 2) // 2
                    if cds_pos < middle:
                        centered_lines.append(line)
                    else:
                        rest_right = max(0, middle - (len(line) - cds_pos))
                        line = line[max(0, cds_pos - middle - rest_right) :]
                        centered_lines.append(line)
            if self.args.encoding not in ["3mer", "5mer"]:
                centered_lines[-1] = self.join_str.join(centered_lines[-1])
        return centered_lines

    def get_only512(self):
        lines = list()
        raw_lines = list()
        for line, raw_line in zip(self.tokenized_seqs, self.lines):
            if len(line) <= self.tokenizer.model_max_length:
                lines.append("".join(line))
                raw_lines.append(raw_line)
        self.lines = raw_lines
        return lines

    def get_3utr(self):
        lines = list()
        raw_lines = list()
        for line, raw_line in zip(self.normalized_lines, self.lines):
            cds_pos = [i for i, x in enumerate(line) if x == "e"]
            if not cds_pos:
                continue
            cds_pos = cds_pos[0]
            line = line[cds_pos + 1 :]
            lines.append(line)
            raw_lines.append(raw_line)
        self.lines = raw_lines
        return lines

    def get_non3utr(self):
        _lines = list()
        for line in self.normalized_lines:
            cds_pos = [i for i, x in enumerate(line) if x == "e"]
            if not cds_pos:
                _lines.append(line)
                continue
            cds_pos = cds_pos[0]
            line = line[:cds_pos]
            _lines.append(line)
        return _lines

    def get_nomarkers(self):
        _lines = list()
        for line in self.normalized_lines:
            line = re.sub("s|e|x", "", line)
            _lines.append(line)
        return _lines

    def log_raw_data(self):
        raw_data_df = pd.DataFrame()
        raw_data_df["seq"] = self.tokenized_seqs
        raw_data_df["lengths"] = raw_data_df["seq"].apply(lambda x: len(x))

        logging.info("Dataset raw statistics:")
        logging.info(raw_data_df.describe(include="all"))

    def log_data(self):
        data_df = pd.DataFrame()
        data_df["seq"] = [
            self.tokenizer.convert_ids_to_tokens(x["input_ids"]) for x in self.examples
        ]
        data_df["lengths"] = data_df["seq"].apply(lambda x: len(x))
        if self.args.mode in ["fine-tune", "predict"]:
            data_df["labels"] = self.labels
        logging.info("Dataset statistics after truncation and adding special tokens:")
        logging.info(data_df.describe(include="all"))

    def _get_OHE(self):
        return None

    @staticmethod
    def tokenize_kmers(lines, args):
        """
        This method is also called when training tokenizers with `learn_tokenizer.py`,
        so we make it static.
        """
        split_lines = list()
        if args.encoding == "3mer":
            pattern = "s|[^xs]{3}|[^xs]{2}x[^xs]|[^xs]x[^xs]{2}|x"
        else:
            pattern = "s|[^xs]{5}|[^xs]{4}x[^xs]|[^xs]x[^xs]{4}||[^xs]{2}x[^xs]{3}|[^xs]{3}x[^xs]{2}|x"
        for line in lines:
            for k, v in eval(args.atomicreplacements).items():
                if args.tokensep is not None:
                    line.replace(
                        f"{args.tokensep}{k}{args.tokensep}",
                        f"{args.tokensep}{v}{args.tokensep}",
                    )
                    line.replace(f"\n{k}{args.tokensep}", f"\n{v}{args.tokensep}")
                    line.replace(f"{args.tokensep}{k}\n", f"{args.tokensep}{v}\n")
                else:
                    line.replace(k, v)
            cds_end_pos = [i for i, x in enumerate(line) if x == args.centertoken]
            if not cds_end_pos:
                split_lines.append(re.findall(pattern, line))
                continue
            else:
                cds_end_pos = cds_end_pos[0]
                front = line[:cds_end_pos]
                back = line[cds_end_pos + 1 :]
                split_front = re.findall(pattern, front[::-1])[::-1]
                split_front = [x[::-1] for x in split_front]
                split_back = re.findall(pattern, back)
                split_line = split_front + ["s"] + split_back
                split_lines.append(split_line)
        return split_lines
