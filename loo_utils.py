import copy

import numpy as np
from ferret.evaluators.faithfulness_measures import TauLOO_Evaluation
from ferret.model_utils import ModelHelper

lowers = ["a", "c", "g", "t"]
uppers = ["A", "C", "G", "T"]
lower_js = ["aej", "cej", "gej", "tej"]
upper_js = ["AEJ", "CEJ", "GEJ", "TEJ"]

replacement_lists = [lowers, uppers, lower_js, upper_js]


class TauLOO_Evaluation_For_Regression(TauLOO_Evaluation):

    def __init__(self, model, tokenizer, OHE=None, tokensep=None):
        self.helper = ModelHelper(model, tokenizer)
        self.OHE = OHE
        self.tokensep = tokensep

    def compute_leave_one_out_occlusion(
        self,
        text,
        target=1,
        remove_first_last=True,
        handle_tokens="remove",
        scaler=None,
        batch_size=8,
    ):

        _, logits = self.helper._forward(
            text,
            self.OHE,
            output_hidden_states=False,
            add_special_tokens=self.OHE is None,
            batch_size=batch_size,
        )
        if logits.size(-1) == 1:
            pred = logits[:, target].cpu().numpy()
        else:
            pred = logits.softmax(-1)[:, target].cpu().numpy()

        item = self.helper._tokenize(text, add_special_tokens=self.OHE is None)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last == True and self.OHE is None:
            input_ids = input_ids[1:-1]

        samples = list()
        if self.OHE is not None:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        else:
            tokens = self.OHE.inverse_transform(input_ids)
            tokens = [x[0] for x in tokens if x != [None]]
            tokens = self.tokenizer.convert_ids_to_tokens(tokens)

        replacements = None
        if handle_tokens == "replace":
            replacements = list()

        for occ_idx in range(len(input_ids)):
            sample = copy.copy(input_ids)
            if handle_tokens == "replace":
                occ_token = self.tokenizer.convert_ids_to_tokens(sample[occ_idx])
                replace_list = [l for l in replacement_lists if occ_token in l][0]
                replace_list = [x for x in replace_list if x != occ_token]
                replacements.append(replace_list)
                replace_list = [
                    self.tokenizer.convert_tokens_to_ids(x) for x in replace_list
                ]
                for r in replace_list:
                    sample[occ_idx] = r
                    decoded_sample = self.tokenizer.decode(sample)
                    # Sadly, we have to do this, as the tokenization with a token separator is not
                    # invertible by decoding.
                    if self.tokensep is not None:
                        decoded_sample = decoded_sample.replace(" ", self.tokensep)
                    samples.append(decoded_sample)
            else:
                if handle_tokens == "remove":
                    sample.pop(occ_idx)
                elif handle_tokens == "mask":
                    sample[occ_idx] = self.tokenizer.mask_token_id

                sample = self.tokenizer.decode(sample)
                samples.append(sample)

        _, logits = self.helper._forward(
            samples,
            self.OHE,
            output_hidden_states=False,
            add_special_tokens=self.OHE is None,
            batch_size=batch_size,
        )
        if logits.size(-1) == 1:
            leave_one_out_removal = logits[:, target].cpu()
        else:
            leave_one_out_removal = logits.softmax(-1)[:, target].cpu()

        if scaler is None:
            occlusion_importance = leave_one_out_removal - pred
        else:
            pred = scaler.inverse_transform(pred)
            leave_one_out_removal = scaler.inverse_transform(leave_one_out_removal)
            occlusion_importance = leave_one_out_removal - pred

        return occlusion_importance, pred, tokens, replacements
