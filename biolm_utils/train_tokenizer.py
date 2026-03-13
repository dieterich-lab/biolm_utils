import random
import tempfile
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Replace
from tokenizers.normalizers import Sequence as Normseq
from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit
from tokenizers.processors import BertProcessing

from biolm_utils.entry import TOKENIZERFILE, logging
from biolm_utils.rna_datasets import RNABaseDataset

# UNIREFSIZE = 152_670_23

def tokenize(args):
    file_path = Path(args.filepath)
    if args.samplesize is not None:
        sample_file_path = (
            file_path.parent / (file_path.stem + f"_{args.samplesize}_samples")
        ).with_suffix(file_path.suffix)

        with open(args.filepath) as f:
            newlines = [f.tell()]
            line = f.readline()
            while line:
                newlines.append(f.tell())
                line = f.readline()
            random.seed(0)
            sample_new_lines = random.sample(newlines, args.samplesize)
            sample_lines = list()
            for l in sorted(sample_new_lines):
                f.seek(l)
                line = f.readline()
                sample_lines.append(line.strip())

        with open(sample_file_path, "w") as sample_file:
            sample_file.write("\n".join(sample_lines))

        file_path = sample_file_path

    if args.encoding == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    elif args.encoding == "atomic":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # normalization and pre-encoding.
    tok_seq = list()
    if args.encoding not in ["3mer", "5mer"]:
        # Normalization for Byte Pair Encoding
        norm_seq = list()

        # Replace multi-char markers by ASCII character.
        if args.atomicreplacements is not None:
            for k, v in eval(args.atomicreplacements).items():
                if args.tokensep is not None:
                    norm_seq.append(
                        Replace(
                            f"{args.tokensep}{k}{args.tokensep}",
                            f"{args.tokensep}{v}{args.tokensep}",
                        )
                    )
                    norm_seq.append(
                        Replace(
                            f"{args.tokensep}{k}{args.tokensep}",
                            f"{args.tokensep}{v}{args.tokensep}",
                        )
                    )
                    # We need another expression to take care of the first and last token.
                    norm_seq.append(
                        Replace(f"\n{k}{args.tokensep}", f"\n{v}{args.tokensep}")
                    )
                    norm_seq.append(
                        Replace(f"{args.tokensep}{k}\n", f"{args.tokensep}{v}\n")
                    )
                else:
                    norm_seq.append(Replace(k, v))

        # Replace the specific information.
        if args.specifiersep is not None:
            norm_seq.append(
                Replace(Regex(rf"{args.specifiersep}[^{args.tokensep}]*"), "")
            )

        # Now join the lines based on the token separator.
        if args.encoding == "bpe":
            if args.tokensep is not None:
                norm_seq.append(Replace(args.tokensep, ""))
            tok_seq.append(pre_tokenizers.ByteLevel(add_prefix_space=True))
            tokenizer.decoder = decoders.ByteLevel()
        elif args.encoding == "atomic":
            if args.tokensep is not None:
                norm_seq.append(Replace(args.tokensep, " "))
            else:
                tok_seq.append(Split(pattern=Regex("."), behavior="isolated"))

            tok_seq.append(WhitespaceSplit())
        norm_seq.append(Replace('"', ""))
        tokenizer.normalizer = Normseq(norm_seq)
    elif args.encoding in ["3mer", "5mer"]:
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
            sample_lines = [x.split(args.columnsep)[-1] for x in sample_lines]
            # TODO: adapt this with to new separation options
            split_lines = RNABaseDataset.tokenize_kmers(sample_lines, args)

    # The list of tokenizer steps.
    pre_seq = list()

    # splitting lines
    pre_seq.append(Split(pattern="\n", behavior="removed"))

    # removing metadata left
    pre_seq.append(
        Split(
            pattern=Regex(
                f"^([^{args.columnsep}]*{args.columnsep}){{{int(args.seqpos) - 1}}}"
            ),
            behavior="removed",
        )
    )
    # removing metadata right
    pre_seq.append(Split(pattern=Regex(f"{args.columnsep}.*"), behavior="removed"))

    # Create an actual pre-tokenization sequence.
    seq = Sequence(pre_seq + tok_seq)
    tokenizer.pre_tokenizer = seq

    # We use the same special tokens as in BERT, no matter what model we actually use.
    SPECIALTOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]

    # Either use Byte-Pair Encoding or a whitespace encoder (for the k-mers).
    if args.encoding == "bpe":
        trainer = trainers.BpeTrainer(
            min_frequency=args.minfreq,
            max_token_length=args.maxtokenlength,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIALTOKENS,
        )
    else:
        trainer = trainers.WordLevelTrainer(
            special_tokens=SPECIALTOKENS,
            vocab_size=args.vocabsize,
            min_frequency=args.minfreq,
        )

    if args.encoding in ["3mer", "5mer"]:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write("\n".join([" ".join(x) for x in split_lines]).encode())
            logging.info(f"Tokenizing {file_path} with temp file {tmp.name}")
            tokenizer.train([tmp.name], trainer)
    else:
        logging.info(f"Tokenizing {file_path}")
        tokenizer.train([str(file_path)], trainer)

    # Add standard BERT post-processing.
    tokenizer.post_processor = BertProcessing(
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
        cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
    )

    tokenizer.name_or_path = TOKENIZERFILE

    # Save the tokenizer.
    logging.info(f"Saving tokenizer to {TOKENIZERFILE}")
    tokenizer.save(str(TOKENIZERFILE))
