import logging
import pickle
import warnings

import pandas as pd
import torch

# see: https://github.com/shap/shap/issues/2909
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap

from loo_utils import TauLOO_Evaluation_For_Regression
from train_utils import get_model


def loo_scores(
    args, tokenizer, model_cls, test_dataset, model_load_path, remove_first_last
):
    pkl_file = model_load_path / f"loo_scores_{args.handletokens}.pkl"
    csv_file = model_load_path / f"loo_scores_{args.handletokens}.csv"

    test_idx = test_dataset.indices

    # Getting the model config.
    config = model_cls.get_config(args, tokenizer, args.blocksize)

    # Determine the output size of the network.
    nlabels = 1

    # Getting model.
    model = get_model(
        args, model_cls, tokenizer, config, model_load_path, nlabels, test_dataset
    )

    # Send the model to the proper device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loo_scores = list()

    tl = TauLOO_Evaluation_For_Regression(
        model, tokenizer, OHE=test_dataset.dataset.OHE, tokensep=args.tokensep
    )

    # for i, id in enumerate(test_idx[:1]):
    for i, test_id in enumerate(test_idx):
        logging.info(f"{i}, {test_id}")
        seq = test_dataset.dataset.lines[test_id]
        loo_score, rescaled_pred, tokens, replacements = (
            tl.compute_leave_one_out_occlusion(
                seq,
                target=0,
                remove_first_last=remove_first_last,
                handle_tokens=args.handletokens,
                scaler=test_dataset.dataset.scaler,
                batch_size=args.batchsize,
            )
        )
        tokens = [x.replace("Ġ", "") for x in tokens]
        scores = list()
        if replacements is not None:
            for j, (token, replacement) in enumerate(zip(tokens, replacements)):
                for k, rep in enumerate(replacement):
                    scores.append(
                        (token, rep, loo_score[j + k].item(), rescaled_pred, j)
                    )
            loo_scores.append(scores)
        else:
            # Here, we'll save the scores conveniently as `shap.Explanation` object.
            exp = shap.Explanation(
                loo_score[None, :], rescaled_pred, feature_names=tokens
            )
            loo_scores.append(exp)

        label = test_dataset.dataset[test_id]["labels"]
        rescaled_label = test_dataset.dataset.scaler.inverse_transform(label)
        seq = test_dataset.dataset.seq_idx[test_id]
        if replacements is not None:
            tokens = [x[0] for x in loo_scores[-1]]
            reps = [x[1] for x in loo_scores[-1]]
            loos = [x[2] for x in loo_scores[-1]]
            preds = [x[3].item() for x in loo_scores[-1]]
            poss = [x[4] for x in loo_scores[-1]]
            seqs = [seq] * len(loos)
            labels = [rescaled_label.item()] * len(loos)
        else:
            loos = loo_scores[-1].values.tolist()[0]
            seqs = [seq] * loo_scores[-1].shape[1]
            labels = [rescaled_label.item()] * loo_scores[-1].shape[1]
            tokens = loo_scores[-1].feature_names
            preds = list(loo_scores[-1].base_values) * loo_scores[-1].shape[1]
        starts, ends = list(), list()
        pos = 0
        for k, token in enumerate(tokens):
            if replacements is not None:
                if k == 0:
                    starts.append(0)
                    ends.append(len(token))
                else:
                    if poss[k] == pos:
                        starts.append(starts[-1])
                        ends.append(ends[-1])
                    else:
                        pos = poss[k]
                        starts.append(ends[-1])
                        ends.append(ends[-1] + len(token))
            else:
                if k == 0:
                    starts.append(0)
                    ends.append(len(token))
                else:
                    starts.append(ends[-1])
                    ends.append(ends[-1] + len(token))
        assert len(loos) == len(starts) == len(ends)

        if replacements is not None:
            data = list(
                zip(
                    seqs,
                    tokens,
                    reps,
                    labels,
                    preds,
                    starts,
                    ends,
                    loos,
                )
            )
            columns = [
                "sequence",
                "token",
                "replacement",
                "label",
                "pred",
                "start_offset",
                "end_offset",
                "loo",
            ]
        else:
            data = list(
                zip(
                    seqs,
                    tokens,
                    labels,
                    preds,
                    starts,
                    ends,
                    loos,
                )
            )
            columns = [
                "sequence",
                "token",
                "label",
                "pred",
                "start_offset",
                "end_offset",
                "loo",
            ]
        if i == 0:
            logging.info(f"Saving to {csv_file}")
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(csv_file, index=False)
        else:
            logging.info(f"Adding to {csv_file}")
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False, mode="a", header=False)

    logging.info(f"Saving to {pkl_file}")
    with open(pkl_file, "wb") as f:
        pickle.dump(loo_scores, f)
