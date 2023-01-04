from typing import Sequence, Tuple
import argparse

import torch
from torch import nn
import torch.nn.functional as F


class Args(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def args_parser(self):
        self.parser.add_argument(
            "--num_layers", default=33, type=int, metavar="N", help="number of layers"
        )
        self.parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        self.parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        self.parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        self.parser.add_argument("--max_positions", default=1024, type=int,
                                 help="number of positional embeddings to learn")
        self.parser.add_argument("--emb_layer_norm_before", default=True, type=bool)
        self.parser.add_argument('--num-sequences', type=int, help='Number of sequences to analyze', default=5000)
        self.parser.add_argument('--logit_bias', type=bool, default=True)
        self.parser.add_argument('--checkpoint_path', type=str, help='the path of load models',
                                 default="E:/esm1b_t33_650M_UR50S.pt")
        args = self.parser.parse_args()
        return args


class Convert(object):
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
            sequence_length = len(encoded_sequence)
            encoded_sequence = torch.tensor(encoded_sequence, dtype=torch.int64)
            tokens[i, 0] = self.cls_idx
            tokens[i, 1:len(encoded_sequence) + 1] = encoded_sequence
            tokens[i, len(encoded_sequence) + 1] = self.eos_idx

        return tokens


class SequenceRegressionDecoder(nn.Module):
    """
        Process the embedding of the model into the format required by the downstream task called flip, stability or fluorescence

        Args:
            embed_dim (int, default=1280): embedding dimension
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, 1)

    def forward(self, prev_result, seqs):
        embedding = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype=prev_result.dtype
        )
        for idx, output_sequence in enumerate(prev_result):
            embedding[idx, :] = torch.mean(output_sequence[:len(seqs[idx]) + 2, :], axis=0)
        x = self.dense(embedding)
        x = F.relu(x)
        x = self.classifier(x)
        return x
