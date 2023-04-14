# Examples

This folder contains examples of actively maintained OpenProtein, organized by data processing, pre-training and downstream tasks.

Data processing methods are provided in the `dataprocess` folder, which converts the original data file into a data file that can be read directly by openprotein.

To learn how to load pre-trained models using the openprotein package, you can open the `pre-training` folder. The currently available pre-trained models are listed in the Table of Pre-training Models.

To learn how to load a downstream task model using the openprotein package, you can open the `downstream_tasks` folder. The currently available downstream tasks are shown in the Table of Downstream Tasks.

## Table of Pre-training Models

|    Model    |                                 pre-trained weights                                 |
| :---------: | :---------------------------------------------------------------------------------: |
|    Esm1b    | [Download](https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt) |
|   GearNet   |                     [Download](https://zenodo.org/record/7593637)                     |
| ProteinBert |                  [Download](https://huggingface.co/Rostlab/prot_bert)                  |

## Table of Downstream Tasks

| Task |                                   datasets url                                   |
| :--: | :------------------------------------------------------------------------------: |
|  EC  | [Download](https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-EC.tar.gz) |
| FLIP |           [Download](https://github.com/J-SNACKKB/FLIP/tree/main/splits)           |
| TAPE |              [Download](https://github.com/songlab-cal/tape#lmdb-data)              |
|  GO  | [Download](https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-GO.tar.gz) |

You can download all dataset in [Alibaba Drive](https://www.aliyundrive.com/s/WbhYzXZJpDW) with the extraction code "HICAI".
