import logging
import random
import tempfile
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Replace
from tokenizers.normalizers import Sequence as Normseq
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
from tokenizers.processors import BertProcessing

from .biolm_dataset import RNABaseDataset
from .path_setup import PathsManager

# UNIREFSIZE = 152_670_237


def tokenize(args):
    paths = PathsManager.get_paths()
    file_path = Path(
        getattr(args.data_source, "filepath", getattr(args, "filepath", ""))
    )
    if getattr(args.tokenization, "samplesize", None) is not None:

        sample_file_path = (
            file_path.parent
            / (file_path.stem + f"_{getattr(args.tokenization, 'samplesize')}_samples")
        ).with_suffix(file_path.suffix)

        with open(file_path) as f:
            if getattr(args.data_source, "stripheader", False):
                f.readline() # skip header
            newlines = [f.tell()]
            line = f.readline()
            while line:
                newlines.append(f.tell())
                line = f.readline()
            random.seed(0)
            sample_new_lines = random.sample(
                newlines, getattr(args.tokenization, "samplesize")
            )
            sample_lines = list()
            for l in sorted(sample_new_lines):
                f.seek(l)
                line = f.readline()
                sample_lines.append(line.strip())

        with open(sample_file_path, "w") as sample_file:
            sample_file.write("\n".join(sample_lines))

        file_path = sample_file_path

    encoding = getattr(args.tokenization, "encoding", "atomic")
    tokensep = getattr(args.data_source, "tokensep", None)
    specsep = getattr(args.data_source, "specifiersep", None)
    if encoding == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    elif encoding == "atomic":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # normalization and pre-encoding.
    tok_seq = list()
    if encoding not in ["3mer", "5mer"]:
        # Normalization for Byte Pair Encoding
        norm_seq = list()

        # Replace multi-char markers by ASCII character.
        from .tokenization_helpers import parse_atomic_replacements

        rep = parse_atomic_replacements(
            getattr(args.tokenization, "atomicreplacements", None)
        )
        if rep is not None:
            for k, v in rep.items():
                tokensep = getattr(args.data_source, "tokensep", None)
                if tokensep is not None:
                    norm_seq.append(
                        Replace(f"{tokensep}{k}{tokensep}", f"{tokensep}{v}{tokensep}")
                    )
                    norm_seq.append(
                        Replace(f"{tokensep}{k}{tokensep}", f"{tokensep}{v}{tokensep}")
                    )
                    # We need another expression to take care of the first and last token.
                    norm_seq.append(Replace(f"\n{k}{tokensep}", f"\n{v}{tokensep}"))
                    norm_seq.append(Replace(f"{tokensep}{k}\n", f"{tokensep}{v}\n"))
                else:
                    norm_seq.append(Replace(k, v))

        # Replace the specific information.
        specsep = getattr(args.data_source, "specifiersep", None)
        if specsep is not None:
            tokensep = getattr(args.data_source, "tokensep", None)
            norm_seq.append(Replace(Regex(rf"{specsep}[^{tokensep}]*"), ""))

        # Now join the lines based on the token separator.
        if encoding == "bpe":
            if tokensep is not None:
                norm_seq.append(Replace(tokensep, ""))
            tok_seq.append(pre_tokenizers.ByteLevel(add_prefix_space=True))
            tokenizer.decoder = decoders.ByteLevel()
        elif encoding == "atomic":
            if tokensep is not None:
                norm_seq.append(Replace(tokensep, " "))
            else:
                tok_seq.append(Split(pattern=Regex("."), behavior="isolated"))

            tok_seq.append(WhitespaceSplit())
        norm_seq.append(Replace('"', ""))

        tokenizer.normalizer = Normseq(norm_seq)
    elif encoding in ["3mer", "5mer"]:
        # The 3mer/5mer processing is too complex to be implemented with the tokenizer regex patterns.
        # We therefore open the file, process the k-merization with regular regex patterns and write the results to a temporary file.
        # The actual tokenizer is then just a white space tokenizer.
        tok_seq.append(WhitespaceSplit())
        with open(file_path, encoding="utf-8") as f:
            sample_lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
            if getattr(args.data_source, "stripheader", False) and sample_lines:
                sample_lines = sample_lines[1:]
            sample_lines = [
                x.split(getattr(args.data_source, "columnsep", "\t"))[-1]
                for x in sample_lines
            ]
            # TODO: adapt this with to new separation options
            split_lines = RNABaseDataset.tokenize_kmers(sample_lines, args)

    # The list of tokenizer steps.
    pre_seq = list()

    # splitting lines
    pre_seq.append(Split(pattern="\n", behavior="removed"))

    # removing header if needed (for BPE/atomic encoding, the file is read by the HuggingFace Tokenizer)
    if getattr(args.data_source, "stripheader", False) and encoding not in ["3mer", "5mer"]:
        # The first token is the header. Since the tokenizer doesn't have an easy way to drop the first line after training,
        # we can't easily strip the header during the `train()` call if it just reads from the file.
        # Actually, replacing the file with a temp file that doesn't have a header is the safest approach.
        pass

    # removing metadata left
    colsep = getattr(args.data_source, "columnsep", "\t")
    # how many tokens to skip (seqpos defaults to 1; we need count - 1)
    seqpos_value = getattr(getattr(args, "data_source", None), "seqpos", None)
    if seqpos_value is None:
        seqpos_value = 1
    seqpos_count = int(seqpos_value) - 1
    pattern_left = f"^([^{colsep}]*{colsep})" + "{" + str(seqpos_count) + "}"
    pre_seq.append(Split(pattern=Regex(pattern_left), behavior="removed"))
    # removing metadata right
    pattern_right = f"{colsep}.*"
    pre_seq.append(Split(pattern=Regex(pattern_right), behavior="removed"))

    # Create an actual pre-tokenization sequence.
    seq = Sequence(pre_seq + tok_seq)
    tokenizer.pre_tokenizer = seq

    # We use the same special tokens as in BERT, no matter what model we actually use.
    SPECIALTOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

    # Either use Byte-Pair Encoding or a whitespace encoder (for the k-mers).
    if encoding == "bpe":
        trainer = trainers.BpeTrainer(
            min_frequency=getattr(args.tokenization, "minfreq", 2),
            max_token_length=getattr(args.tokenization, "maxtokenlength", 10),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIALTOKENS,
        )
    else:
        trainer = trainers.WordLevelTrainer(
            special_tokens=SPECIALTOKENS,
            vocab_size=getattr(args.tokenization, "vocabsize", 20000),
            min_frequency=getattr(args.tokenization, "minfreq", 2),
        )

    if encoding in ["3mer", "5mer"]:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write("\n".join([" ".join(x) for x in split_lines]).encode())
            logging.info(f"Tokenizing {file_path} with temp file {tmp.name}")
            tokenizer.train([tmp.name], trainer)
    else:
        if getattr(args.data_source, "stripheader", False):
            with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
                with open(file_path, encoding="utf-8") as orig_f:
                    lines = orig_f.readlines()
                    if lines:
                        tmp.writelines(lines[1:])
                tmp_name = tmp.name
            logging.info(f"Tokenizing {file_path} with header stripped via temp file {tmp_name}")
            tokenizer.train([tmp_name], trainer)
            import os
            os.remove(tmp_name)
        else:
            logging.info(f"Tokenizing {file_path}")
            tokenizer.train([str(file_path)], trainer)

    # Add standard BERT post-processing.
    tokenizer.post_processor = BertProcessing(
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
        cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
    )

    tokenizer.name_or_path = paths["TOKENIZERFILE"]

    # Save the tokenizer in HuggingFace format for compatibility with PreTrainedTokenizerFast
    logging.info(f"Saving tokenizer to {paths['TOKENIZERFILE']}")

    # Convert to HuggingFace format by creating a PreTrainedTokenizerFast
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        mask_token="[MASK]",
        cls_token="[CLS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=(
            getattr(args.training, "blocksize", 12288)
            if hasattr(args, "training") and args.training
            else 12288
        ),
        truncation=True,
        truncation_side=(
            "left"
            if getattr(getattr(args, "tokenization", None), "lefttailing", False)
            else "right"
        ),
    )

    # Save in HuggingFace format (creates directory with tokenizer.json, tokenizer_config.json, etc.)
    tokenizer_dir = paths["TOKENIZERFILE"]
    hf_tokenizer.save_pretrained(str(tokenizer_dir))
