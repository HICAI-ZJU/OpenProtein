from dataclasses import dataclass, field


@dataclass
class Esm1bConfig:
    num_layers: int = field(
        default=33
    )
    embed_dim: int = field(
        default=1280
    )
    logit_bias: bool = field(
        default=True
    )
    ffn_embed_dim: int = field(
        default=5120
    )
    attention_heads: int = field(
        default=20
    )
    max_positions: int = field(
        default=1024
    )
    emb_layer_norm_before: bool = field(
        default=True
    )
    checkpoint_path: str = field(
        default=None
    )


@dataclass
class GearNetConfig:
    input_dim: int = field(
        default=21
    )
    embedding_dim: int = field(
        default=512
    )
    hidden_dims: list = field(
        default_factory=list,
        metadata={"help": "[512, 512, 512, 512, 512, 512]"},
    )
    num_relation: int = field(
        default=7
    )
    edge_input_dim: int = field(
        default=59
    )
    batch_norm: int = field(
        default=True
    )
    activation: bool = field(
        default='relu'
    )
    concat_hidden: bool = field(
        default=True
    )
    short_cut: bool = field(
        default=True
    )
    readout: str = field(
        default="sum"
    )
    dropout: int = field(
        default=0.2
    )
    num_angle_bin: int = field(
        default=8
    )
    layer_norm: bool = field(
        default=True
    )
    use_ieconv: bool = field(
        default=True
    )