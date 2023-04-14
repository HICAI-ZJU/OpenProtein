from typing import *
import itertools
import torch
import numpy as np


PROTEINSEQ = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                'X', 'B', 'U', 'Z', 'O', '.', '-']


class MaskedConverter(object):
    def __init__(self, standard_toks: Sequence[str],
                 prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
                 append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
                 prepend_bos: bool = True,
                 append_eos: bool = False):

        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.protein_tok_begin = len(self.all_toks)
        self.all_toks.extend(self.standard_toks)
        self.protein_tok_end = len(self.all_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>']
        self.unique_no_split_tokens = self.all_toks

        self.mask_prob = 0.15
        self.random_token_prob = 0.1
        self.leave_unmasked_prob = 0.1

        weights = np.zeros(len(self.all_toks))
        weights[:len(self.all_toks) - 2] = 1
        self.weights = weights / weights.sum()

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def __len__(self):
        return len(self.all_toks)

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        batch_size = len(raw_batch)
        encoded_sequences = [self.encode(sequence) for sequence in raw_batch]
        max_encoded_sequences_length = max(len(encoded_sequence) for encoded_sequence in encoded_sequences)
        origin_tokens = torch.empty(
            (
                batch_size,
                max_encoded_sequences_length + 2,
            ),
            dtype=torch.int64
        )
        origin_tokens.fill_(self.padding_idx)
        masked_tokens = torch.empty(
            (
                batch_size,
                max_encoded_sequences_length + 2,
            ),
            dtype=torch.int64
        )
        masked_tokens.fill_(self.padding_idx)
        target_tokens = torch.empty(
            (
                batch_size,
                max_encoded_sequences_length + 2,
            ),
            dtype=torch.int64
        )
        target_tokens.fill_(self.padding_idx)

        for i, encoded_sequence in enumerate(encoded_sequences):
            sequence_length = len(encoded_sequence)
            mask = np.full(sequence_length, False)
            num_mask = int(self.mask_prob * sequence_length + np.random.rand())
            mask_idc = np.random.choice(sequence_length, num_mask, replace=False)
            mask_idc = mask_idc[mask_idc < len(mask)]
            mask[mask_idc] = True

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            rand_or_unmask = mask & (np.random.rand(sequence_length) < rand_or_unmask_prob)
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
            else:
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = np.random.rand(sequence_length) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)

            mask = mask ^ unmask

            masked_sequence = np.copy(encoded_sequence)
            masked_sequence[mask] = self.mask_idx
            num_rand = rand_mask.sum()
            masked_sequence[rand_mask] = np.random.choice(
                len(self.all_toks),
                num_rand,
                p=self.weights,
            )
            encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.int64)
            masked_sequence = torch.tensor(masked_sequence, dtype=torch.int64)
            origin_tokens[i, 0] = self.cls_idx
            origin_tokens[i, 1:len(encoded_sequence) + 1] = encoded_sequence
            origin_tokens[i, len(encoded_sequence) + 1] = self.eos_idx

            masked_tokens[i, 0] = self.cls_idx
            masked_tokens[i, 1:len(masked_sequence) + 1] = masked_sequence
            masked_tokens[i, len(masked_sequence) + 1] = self.eos_idx

            target_tokens[i, 1:len(mask) + 1][mask | unmask] = encoded_sequence[mask | unmask]

        return origin_tokens, masked_tokens, target_tokens

    @classmethod
    def build_convert(cls, proteinseq: dict=None) -> "MaskedConverter":
        standard_toks = proteinseq if proteinseq else PROTEINSEQ
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    def _tokenize(self, text: str) -> str:
        return text.split()

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list: List, text: str):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class TaskConvert(object):
    """
    Converter for tokenizing protein sequence.

    Args:
        alphabet: Dictionary from amino acids to tokens
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    def __call__(self, seqences: Sequence[Tuple[str, str]]):
        """
        Convert Batch to Downstream Task Needed Format.

        Args:
            a batch of proteins [sequence1, sequence2, ...]

        Returns:
            aligned tokens
        """
        batch_size = len(seqences)

        encoded_sequences = [self.alphabet.encode(sequence) for sequence in seqences]
        max_encoded_sequences_length = max(len(encoded_sequence) for encoded_sequence in encoded_sequences)
        tokens = torch.empty(
            (
                batch_size,
                max_encoded_sequences_length + 2,
            ),
            dtype=torch.int64
        )
        tokens.fill_(self.pad_idx)

        for i, encoded_sequence in enumerate(encoded_sequences):
            encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.int64)
            tokens[i, 0] = self.cls_idx
            tokens[i, 1:len(encoded_sequence) + 1] = encoded_sequence
            tokens[i, len(encoded_sequence) + 1] = self.eos_idx

        return tokens


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.protein_tok_begin = len(self.all_toks)
        self.all_toks.extend(self.standard_toks)
        self.protein_tok_end = len(self.all_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def pad(self):
        return self.padding_idx

    @classmethod
    def build_alphabet(cls, proteinseq: dict = PROTEINSEQ) -> "Alphabet":
        standard_toks = proteinseq if proteinseq else PROTEINSEQ
        prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
        append_toks = ("<mask>",)
        prepend_bos = True
        append_eos = True
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]