from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from eo import Transformer


def pad_seq(sequences, batch_first=True, padding_value=0.0, prepadding=True):
    lens = [item.shape[0] for item in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
    if prepadding:
        for index, length in enumerate(lens):
            padded_sequences[index] = padded_sequences[index].roll(-length)
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1)
    return padded_sequences


def get_batches(items, batch_size=16):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class TashkeelModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        d_model=512,
        n_layers=6,
        n_heads=16,
        drop_prob=0.1,
        learnable_pos_emb=False,
    ):
        super().__init__()
        ffn_hidden = 4 * d_model
        src_pad_idx = tokenizer.letters_map["<PAD>"]
        trg_pad_idx = tokenizer.tashkeel_map["<PAD>"]
        enc_voc_size = len(tokenizer.letters_map)
        dec_voc_size = len(tokenizer.tashkeel_map)
        self.transformer = Transformer(
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            d_model=d_model,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_seq_len,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layers=n_layers,
            drop_prob=drop_prob,
            learnable_pos_emb=learnable_pos_emb,
        )
        self.tokenizer = tokenizer

    def forward(self, input_ids):
        return self.transformer(input_ids)

    def force_no_tashkeel_for_non_letters(self, input_ids, pred_ids):
        nt_id = self.tokenizer.tashkeel_map[self.tokenizer.no_tashkeel_tag]
        no_tashkeel_mask = input_ids == self.tokenizer.letters_map[" "]
        for punct in self.tokenizer.punctuation_tokens:
            no_tashkeel_mask |= input_ids == self.tokenizer.letters_map[punct]
        pred_ids[no_tashkeel_mask] = nt_id
        return pred_ids

    @torch.no_grad()
    def do_tashkeel_batch(self, texts, batch_size=16, verbose=False):
        self.eval()
        device = next(self.parameters()).device
        text_with_tashkeel = []
        for texts_mini in get_batches(texts, batch_size):
            input_ids_list = []
            for text in texts_mini:
                input_ids, _ = self.tokenizer.encode(text, test_match=False)
                input_ids_list.append(input_ids)

            batch_input_ids = pad_seq(
                input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.letters_map["<PAD>"],
                prepadding=False,
            )
            batch_input_ids = batch_input_ids[:, 1:-1].to(device)
            y_pred = self(batch_input_ids).argmax(-1)
            y_pred = self.force_no_tashkeel_for_non_letters(batch_input_ids, y_pred)
            text_with_tashkeel.extend(self.tokenizer.decode(batch_input_ids, y_pred))
        return text_with_tashkeel
