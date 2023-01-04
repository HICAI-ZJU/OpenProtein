<h1 align="center">
    <p>Open-Protein</p>
</h1>

Open-Protein is an open source pre-training platform that supports multiple protein pre-training models and downstream tasks.

# Table of Contents
[TOC]

# Qucik Start
This repo is tested on Python 3.6 - 3.9 and PyTorch 1.5.0+ (PyTorch 1.5.0+ for examples).

When you install Open-Protein, we do not force the installation of pytorch dependencies, you are free to choose a version of Pytorch 1.5 or higher.

## With Pip

First you need to install PyTorch and have a version number greater than 1.5.0.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, Open-Protein can be installed using pip as follows:

```bash
pip install openprotein
```

## From source

Here also, you first need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/HIC-AIX/OpenProtein.git
cd open-protein
pip install .
```

When you update the repository, you should upgrade the transformers installation and its dependencies as follows:

```bash
git pull
pip install --upgrade .
```

# Example
```python
import torch
import torch.nn.functional as F

from openprotein.datasets import Uniref
from openprotein.data import MaskedConverter, Alphabet
from openprotein.models import Esm1b
from openprotein.utils import Accuracy
uniref = Uniref("./valid")

proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                'X', 'B', 'U', 'Z', 'O', '.', '-']
}
converter = MaskedConverter.build_convert(proteinseq_toks)
alphabet = Alphabet.build_alphabet(proteinseq_toks)

args = {'num_layers': 33, 'embed_dim': 1280, 'logit_bias': True, 'ffn_embed_dim': 5120,
                'attention_heads': 20,
                'max_positions': 1024, 'emb_layer_norm_before': True, 'checkpoint_path': None}
args = argparse.Namespace(**args)
model = Esm1b(args, alphabet)

f = lambda x: converter(x)
dl = uniref.get_dataloader(collate_fn=f)
for origin_tokens, masked_tokens, target_tokens in dl:
    result = model(masked_tokens)['logits']
    loss = Loss(
        result.contiguous().view(-1, result.size(-1)),
        target_tokens.contiguous().view(-1),
        reduction="mean",
        ignore_index=alphabet.padding_idx
    ).cross_entropy()
    print(loss)
    break
```
