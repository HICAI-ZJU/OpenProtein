# Data processing Examples

This folder contains several use cases for downstream task models. These use cases mainly show the most common ways to call the downstream task models in OpenProtein.

This folder contains data processing code. These codes convert raw data into data files read by openprotein. They are used in the following way.

- Take the EC dataset as an example.

1. Download the ec dataset to the specified folder, e.g. `./data/`

```bash
wget -O ./data/PDB-EC.tar.gz -c https://users.flatironinstitute.org/~renfrew/DeepFRI_data/PDB-EC.tar.gz
```

2. Unzip to the specified folder, e.g. `. /data/ec`

```bash
mkdir ./data/ori_ec
tar -zxvf PDB-EC.tar.gz -C ./data/ori_ec
```

3. Run data processing scripts

```bash
mkdir ./data/ec
python conver_to_lmdb.py ec -p ./data/ori_ec -o ./data/ec
```

Wait a moment and you will get the converted EC dataset.
