import torch
import torch.nn.functional as F
from torch import nn


class SequenceClassificaitonDecoder(nn.Module):
    """
        Process the embedding of the model into the format required by the downstream task called remote_homology

        Args:
            embed_dim (int, default=1280): embedding dimension
    """
    def __init__(self, embed_dim, label_num):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, label_num)

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


class SequenceToSequenceClassificaitonDecoder(nn.Module):
    """
        Process the embedding of the model into the format required by the downstream task called secondary_structure

        Args:
            embed_dim (int, default=1280): embedding dimension
    """
    def __init__(self, embed_dim, label_num=3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, label_num)

    def forward(self, prev_result):
        x = self.dense(prev_result)
        x = F.relu(x)
        x = self.classifier(x)
        return x


class ProteinContactMapDecoder(nn.Module):
    """
        Process the embedding of the model into the format required by the downstream task called proteinnet

        Args:
            embed_dim (int, default=1280): embedding dimension
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * embed_dim, 2))

    def forward(self, prev_result):
        prod = prev_result[:, :, None, :] * prev_result[:, None, :, :]
        diff = prev_result[:, :, None, :] - prev_result[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        outputs = prediction
        return outputs
