from __future__ import annotations

import re

import bw2ar
import torch


FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
TATWEEL = "\u0640"

HARAKAT_PAT = re.compile(
    "[" + "".join([FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SUKUN, SHADDA]) + "]"
)


class TashkeelTokenizer:
    """Punctuation-aware tokenizer for the v4 CATT track.

    Important design choice:
      - Existing token ids from the original tokenizer are preserved.
      - New punctuation symbols are appended at the end of the input vocabulary.
      - This lets us warm-start from the v3 checkpoint and only initialize the
        new punctuation embeddings from scratch.
    """

    def __init__(self):
        base_letters = [
            " ",
            "$",
            "&",
            "'",
            "*",
            "<",
            ">",
            "A",
            "D",
            "E",
            "H",
            "S",
            "T",
            "Y",
            "Z",
            "b",
            "d",
            "f",
            "g",
            "h",
            "j",
            "k",
            "l",
            "m",
            "n",
            "p",
            "q",
            "r",
            "s",
            "t",
            "v",
            "w",
            "x",
            "y",
            "z",
            "|",
            "}",
        ]
        self.punctuation_tokens = ["،", "؛", ":", ".", ",", "!", "?", "؟", "(", ")", "[", "]", "«", "»", '"', "-"]
        self.punctuation_set = set(self.punctuation_tokens)
        self.letters = ["<PAD>", "<BOS>", "<EOS>"] + base_letters + ["<MASK>"] + self.punctuation_tokens

        self.no_tashkeel_tag = "<NT>"
        self.tashkeel_list = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<NT>",
            "<SD>",
            "<SDD>",
            "<SF>",
            "<SFF>",
            "<SK>",
            "<SKK>",
            "F",
            "K",
            "N",
            "a",
            "i",
            "o",
            "u",
            "~",
        ]

        self.tashkeel_map = {c: i for i, c in enumerate(self.tashkeel_list)}
        self.letters_map = {c: i for i, c in enumerate(self.letters)}

        self.inverse_tags = {
            "~a": "<SF>",
            "~u": "<SD>",
            "~i": "<SK>",
            "~F": "<SFF>",
            "~N": "<SDD>",
            "~K": "<SKK>",
        }
        self.tags = {v: k for k, v in self.inverse_tags.items()}
        self.shaddah_last = ["a~", "u~", "i~", "F~", "N~", "K~"]
        self.shaddah_first = ["~a", "~u", "~i", "~F", "~N", "~K"]
        self.tashkeel_chars = ["F", "N", "K", "a", "u", "i", "~", "o"]
        punct = re.escape("".join(self.punctuation_tokens))
        self.keep_re = re.compile(
            rf"[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 {punct}]",
            flags=re.UNICODE,
        )

    def clean_text(self, text):
        text = re.sub("[%s]" % TATWEEL, "", text)
        text = text.replace("ٱ", "ا")
        return " ".join(self.keep_re.sub(" ", text).split())

    def check_match(self, text_with_tashkeel, letter_n_tashkeel_pairs):
        text_with_tashkeel = text_with_tashkeel.strip()
        syn_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
        return syn_text == text_with_tashkeel or syn_text == self.unify_shaddah_position(text_with_tashkeel)

    def unify_shaddah_position(self, text_with_tashkeel):
        for i in range(len(self.shaddah_first)):
            text_with_tashkeel = text_with_tashkeel.replace(self.shaddah_last[i], self.shaddah_first[i])
        return text_with_tashkeel

    def split_tashkeel_from_text(self, text_with_tashkeel, test_match=True):
        text_with_tashkeel = self.clean_text(text_with_tashkeel)
        text_with_tashkeel = bw2ar.transliterate_text(text_with_tashkeel, "ar2bw")
        text_with_tashkeel = text_with_tashkeel.replace("`", "")
        text_with_tashkeel = self.unify_shaddah_position(text_with_tashkeel)

        for tash_char in self.tashkeel_chars:
            text_with_tashkeel = text_with_tashkeel.replace(tash_char * 2, tash_char)

        letter_n_tashkeel_pairs = []
        for i in range(len(text_with_tashkeel)):
            if (
                i < (len(text_with_tashkeel) - 1)
                and text_with_tashkeel[i] not in self.tashkeel_list
                and text_with_tashkeel[i + 1] in self.tashkeel_list
            ):
                if text_with_tashkeel[i + 1] == "~":
                    if i + 2 < len(text_with_tashkeel) and f"~{text_with_tashkeel[i + 2]}" in self.inverse_tags:
                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.inverse_tags[f"~{text_with_tashkeel[i + 2]}"]))
                    else:
                        letter_n_tashkeel_pairs.append((text_with_tashkeel[i], "~"))
                else:
                    letter_n_tashkeel_pairs.append((text_with_tashkeel[i], text_with_tashkeel[i + 1]))
            elif text_with_tashkeel[i] not in self.tashkeel_list:
                letter_n_tashkeel_pairs.append((text_with_tashkeel[i], self.no_tashkeel_tag))

        if test_match:
            assert self.check_match(text_with_tashkeel, letter_n_tashkeel_pairs)

        return [("<BOS>", "<BOS>")] + letter_n_tashkeel_pairs + [("<EOS>", "<EOS>")]

    def combine_tashkeel_with_text(self, letter_n_tashkeel_pairs):
        combined_with_tashkeel = []
        for letter, tashkeel in letter_n_tashkeel_pairs:
            combined_with_tashkeel.append(letter)
            if tashkeel in self.tags:
                combined_with_tashkeel.append(self.tags[tashkeel])
            elif tashkeel != self.no_tashkeel_tag:
                combined_with_tashkeel.append(tashkeel)
        return "".join(combined_with_tashkeel)

    def encode(self, text_with_tashkeel, test_match=True):
        letter_n_tashkeel_pairs = self.split_tashkeel_from_text(text_with_tashkeel, test_match)
        text, tashkeel = zip(*letter_n_tashkeel_pairs)
        input_ids = [self.letters_map[c] for c in text]
        target_ids = [self.tashkeel_map[c] for c in tashkeel]
        return torch.LongTensor(input_ids), torch.LongTensor(target_ids)

    def filter_tashkeel(self, tashkeel):
        filtered = []
        for i, item in enumerate(tashkeel):
            if i != 0 and item == "<BOS>":
                item = self.no_tashkeel_tag
            elif i != (len(tashkeel) - 1) and item == "<EOS>":
                item = self.no_tashkeel_tag
            filtered.append(item)
        return filtered

    def decode(self, input_ids, target_ids):
        input_ids = input_ids.cpu().tolist()
        target_ids = target_ids.cpu().tolist()
        ar_texts = []
        for j in range(len(input_ids)):
            letters = [self.letters[i] for i in input_ids[j]]
            tashkeel = [self.tashkeel_list[i] for i in target_ids[j]]

            letters = [x for x in letters if x not in {"<BOS>", "<EOS>", "<PAD>"}]
            tashkeel = self.filter_tashkeel(tashkeel)
            tashkeel = [x for x in tashkeel if x not in {"<BOS>", "<EOS>", "<PAD>"}]

            letter_n_tashkeel_pairs = list(zip(letters, tashkeel))
            bw_text = self.combine_tashkeel_with_text(letter_n_tashkeel_pairs)
            ar_texts.append(bw2ar.transliterate_text(bw_text, "bw2ar"))
        return ar_texts

    def _boundary_indices(self, text_split):
        indices = set()
        for i, (letter, _) in enumerate(text_split):
            if letter in self.punctuation_set or letter in {" ", "<EOS>"}:
                indices.add(i)
        return indices

    def get_tashkeel_with_case_ending(self, text, case_ending=True):
        text_split = self.split_tashkeel_from_text(text, test_match=False)
        boundary_indices = self._boundary_indices(text_split)
        new_text_split = []
        for i, (letter, tashkeel) in enumerate(text_split):
            if letter in self.punctuation_set:
                continue
            if not case_ending and letter not in {"<BOS>", "<EOS>", " "} and (i + 1) in boundary_indices:
                tashkeel = self.no_tashkeel_tag
            new_text_split.append((letter, tashkeel))
        letters, tashkeel = zip(*new_text_split)
        return letters, tashkeel

    def compute_der(self, ref, hyp, case_ending=True):
        import xer

        _, ref_tashkeel = self.get_tashkeel_with_case_ending(ref, case_ending=case_ending)
        _, hyp_tashkeel = self.get_tashkeel_with_case_ending(hyp, case_ending=case_ending)
        return xer.wer(" ".join(ref_tashkeel), " ".join(hyp_tashkeel))

    def compute_wer(self, ref, hyp, case_ending=True):
        import xer

        ref_letters, ref_tashkeel = self.get_tashkeel_with_case_ending(ref, case_ending=case_ending)
        hyp_letters, hyp_tashkeel = self.get_tashkeel_with_case_ending(hyp, case_ending=case_ending)
        ref_text_combined = self.combine_tashkeel_with_text(zip(ref_letters, ref_tashkeel))
        hyp_text_combined = self.combine_tashkeel_with_text(zip(hyp_letters, hyp_tashkeel))
        return xer.wer(ref_text_combined, hyp_text_combined)

    def remove_tashkeel(self, text):
        text = HARAKAT_PAT.sub("", text)
        text = re.sub(u"[\u064E]", "", text, flags=re.UNICODE)
        text = re.sub(u"[\u0671]", "", text, flags=re.UNICODE)
        return text
