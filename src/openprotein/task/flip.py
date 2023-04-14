from typing import Sequence, Tuple
import argparse

import torch
from torch import nn
import torch.nn.functional as F


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
