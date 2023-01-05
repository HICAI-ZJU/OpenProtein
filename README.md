<h1 align="center">
    <p>Open-Protein</p>
</h1>

Open-Protein is an open source pre-training platform that supports multiple protein pre-training models and downstream tasks.


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

# Content
| model  | introduction                                                 | paper                                                        |
| :----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| esm1b  | a 650M-parameter protein language model                      | Genome-wide prediction of disease variants with a deep protein language [paper](https://www.biorxiv.org/content/10.1101/2022.08.25.505311v1.abstract) |
| EC     | a numerical classification scheme for enzymes, based on the chemical reactions they catalyze. As a system of enzyme nomenclature, every EC number is associated with a recommended name for the corresponding enzyme-catalyzed reaction | Structure-based protein function prediction using graph convolutional networks [paper](https://www.nature.com/articles/s41467-021-23303-9) |
| FLIP   | a benchmark for function prediction to encourage rapid scoring of representation learning for protein engineering | FLIP: Benchmark tasks in fitness landscape inference for proteins [paper](https://openreview.net/forum?id=p2dMLEwL8tF) |
| TAPE   | a set of five biologically relevant semi-supervised learning tasks spread across different domains of protein biology | Evaluating Protein Transfer Learning with TAPE [paper](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html) |
| UniRef | The UniRef (UniProt Reference Clusters) provide clustered sets of sequences from the UniProt Knowledgebase (UniProtKB) and selected UniProt Archive records to obtain complete coverage of sequence space at several resolutions while hiding redundant sequences. | UniRef: comprehensive and non-redundant UniProt reference [paper](https://academic.oup.com/bioinformatics/article/23/10/1282/197795?login=false) |
| GO     | GO (gene ontology) is a database established by the Gene Ontology  Consortium. It aims to establish a language vocabulary standard that is  applicable to various species, defines and describes the functions of  genes and proteins, and can be updated with the deepening of research. GO is one of a variety of biological ontology languages, which provides a three-layer system definition method to describe the function of gene products. | Gene Ontology: tool for the unification of biology [paper](https://www.nature.com/articles/ng0500_25) |

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
