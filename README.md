<h1 align="center">
    <p>Open-Protein</p>
</h1>

Open-Protein is an open source pre-training platform that supports multiple protein pre-training models and downstream tasks.

# Qucik Start

This repo is tested on Python 3.6 - 3.9 and PyTorch 1.5.0+ (PyTorch 1.5.0+ for examples).

When you install Open-Protein, we do not force the installation of pytorch dependencies, you are free to choose a version of Pytorch 1.5 or higher.

## With Pip

First you need to install `PyTorch` and `torch-scatter`, where the version number of torch should be greater than 1.5.0, and the version number of torch-scatter should be greater than 2.0.8.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, Open-Protein can be installed using pip as follows:

```bash
pip install openprotein
```

## From source

First you need to install `PyTorch` and `torch-scatter`, where the version number of torch should be greater than 1.5.0, and the version number of torch-scatter should be greater than 2.0.8.
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

## Dataset

You can download the corresponding datasets according to the downstream tasks. The links for each dataset are listed below.

|                     Name                     | url                                                                              |
| :-------------------------------------------: | -------------------------------------------------------------------------------- |
| [Uniref](https://www.uniprot.org/help/downloads) | [Download](https://www.uniprot.org/help/downloads)                                  |
|                      EC                      | [Download](https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-EC.tar.gz) |
|                     Flip                     | [Download](https://github.com/J-SNACKKB/FLIP/tree/main/splits)                      |
|                     Tape                     | [Download](https://github.com/songlab-cal/tape#lmdb-data)                           |
|                      Go                      | [Download](https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-GO.tar.gz) |

## Model

|    model    | introduction                                                                                                          | paper                                                                                                                                                          | github                                                                                                                                                                                                                                                                  | weight                                                                                                                                                |
| :---------: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
|    Esm1b    | A 650M-parameter protein language model.                                                                              | [Genome-wide prediction of disease variants with a deep protein language](https://www.biorxiv.org/content/10.1101/2022.08.25.505311v1.abstract)                   | [rmrao/esm-1: Evolutionary Scale Modeling (esm): Pretrained language models for proteins](https://github.com/rmrao/esm-1)                                                                                                                                                  | [https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt) |
|   GearNet   | Geometry-Aware Relational Graph Neural Network.                                                                       | [GearNet: Stepwise Dual Learning for Weakly Supervised Domain Adaptation](https://personal.ntu.edu.sg/boan/papers/AAAI22-Adaptation.pdf)                          | [GitHub - DeepGraphLearning/GearNet: GearNet and Geometric Pretraining Methods for Protein Structure Representation Learning, ICLR](https://github.com/DeepGraphLearning/GearNet)                                                                                          | [Pre-Trained Models for GearNet](https://zenodo.org/record/7593637)                                                                                      |
| ProteinBert | ProtBert is based on Bert model which pretrained on a large corpus of protein sequences in a self-supervised fashion. | [ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Learning](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3.full.pdf) | [GitHub - agemagician/ProtTrans: ProtTrans is providing state of the art pretrained language models for proteins. ProtTrans was trained on thousands of GPUs from Summit and hundreds of Google TPUs using Transformers Models.](https://github.com/agemagician/ProtTrans) | [Rostlab/prot_bert · Hugging Face](https://huggingface.co/Rostlab/prot_bert)                                                                            |

## Downstream Task

| Task | introduction                                                                                                                                                                                                                                                                                                                                                                                                                                        | paper                                                                                                                                        | dataset                                                                 |
| :--: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
|  EC  | a numerical classification scheme for enzymes, based on the chemical reactions they catalyze. As a system of enzyme nomenclature, every EC number is associated with a recommended name for the corresponding enzyme-catalyzed reaction.                                                                                                                                                                                                            | [Structure-based protein function prediction using graph convolutional networks](https://www.nature.com/articles/s41467-021-23303-9)            | https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-EC.tar.gz |
| FLIP | a benchmark for function prediction to encourage rapid scoring of representation learning for protein engineering.                                                                                                                                                                                                                                                                                                                                  | [FLIP: Benchmark tasks in fitness landscape inference for proteins](https://openreview.net/forum?id=p2dMLEwL8tF)                                | https://github.com/J-SNACKKB/FLIP/tree/main/splits                      |
| TAPE | a set of five biologically relevant semi-supervised learning tasks spread across different domains of protein biology.                                                                                                                                                                                                                                                                                                                              | [Evaluating Protein Transfer Learning with TAPE](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html) | https://github.com/songlab-cal/tape#lmdb-data                           |
|  GO  | GO (gene ontology) is a database established by the Gene Ontology  Consortium. It aims to establish a language vocabulary standard that is  applicable to various species, defines and describes the functions of  genes and proteins, and can be updated with the deepening of research. GO is one of a variety of biological ontology languages, which provides a three-layer system definition method to describe the function of gene products. | [Gene Ontology: tool for the unification of biology](https://www.nature.com/articles/ng0500_25)                                                 | https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-GO.tar.gz |

# Usage

## Data Process

- Take the EC dataset as an example.

```bash
 # 1. Download ec dataset.
wget -O ./data/PDB-EC.tar.gz -c https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-EC.tar.gz

 # 2. Create a new folder for the ec dataset.
mkdir ./data/ori_ec

 # 3. Unzip the ec dataset to the new folder.
tar -zxvf PDB-EC.tar.gz -C ./data/ori_ec

 # 4. Executing dataset processing files.
python conver_to_lmdb.py ec -p ./data/ori_ec -o ./data/ec
```

## Feature Extraction

```python
from openprotein import Esm1b, Esm1bConfig
from openprotein.data import MaskedConverter, Alphabet

seq = 'RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE'

converter = MaskedConverter.build_convert()
alphabet = Alphabet.build_alphabet()

args = Esm1bConfig()

origin_tokens, masked_tokens, target_tokens = converter(seq)
model = Esm1b(args, alphabet)

feature = model(masked_tokens)
```

## Downstream Task

```python
from openprotein import Esm1b, Esm1bConfig
from openprotein.data import MaskedConverter, Alphabet, TaskConvert
from openprotein.task import ProteinFunctionDecoder

seq = 'RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE'

converter = MaskedConverter.build_convert()
alphabet = Alphabet.build_alphabet()
args = Esm1bConfig(checkpoint_path="./resources/esm1b/esm1b_t33_650M_UR50S.pt")

origin_tokens, masked_tokens, target_tokens = converter(seq)
model = Esm1b.load(args, alphabet).eval()

feature = model(masked_tokens)

converter = TaskConvert(alphabet)
protein_function_decoder = ProteinFunctionDecoder(args.embed_dim, args.class_num)
outputs = protein_function_decoder(feature, seq)
print(outputs)
```

# License

This source code is licensed under the OSL 3.0 license found in the `LICENSE` file in the root directory of this source tree
