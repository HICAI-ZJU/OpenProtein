import torch
from torch import nn
import torch.nn.functional as F


class ProteinFunctionDecoder(nn.Module):
    """
        Process the embedding of the model into the format required by the downstream task called ec or go

        Args:
            embed_dim (int, default=1280): embedding dimension
    """
    def __init__(self, embed_dim, class_num):
        super().__init__()
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.dense = nn.Linear(self.embed_dim, int(self.embed_dim / 2))
        self.classifier = nn.Linear(int(self.embed_dim / 2), class_num * 2)

    def forward(self, prev_result, seqs):
        batch_size = prev_result.size(0)
        embedding = torch.empty(
            (prev_result.size(0), prev_result.size(-1)),
            dtype=prev_result.dtype
        )
        for idx, output_sequence in enumerate(prev_result):
            embedding[idx, :] = torch.mean(output_sequence[:len(seqs[idx])+2, :], axis=0)
        x = self.dense(embedding)
        x = F.relu(x)
        x = self.classifier(x).reshape(batch_size, self.class_num, 2)
        return x