from typing import Tuple
import os
import csv
import numpy as np


def load_EC_annot(filename):
    prot2annot = {}
    with open(os.path.join(filename, 'nrPDB-EC_2020.04_annot.tsv'), mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1
    return prot2annot, ec_numbers, ec_numbers, counts


def load_name(filename, split):
    name = []
    if split == 'valid':
        split = 'test'
    with open(os.path.join(filename, f'nrPDB-EC_2020.04_{split}.txt'), 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            name.append(row[0])
    return name


def load_name2sequence(filename):
    name2sequence = {}
    with open(os.path.join(filename, f'nrPDB-EC_2020.04_sequences.fasta'), 'r') as f:
        for line in f:
            if line.startswith('>'):
                name = line.replace('>', '').split()[0]
                name2sequence[name] = ''
            else:
                name2sequence[name] += line.replace('\n', '').strip()
    return name2sequence


class ECFunctionDataset:
    def __init__(self, data_path, task_type, split) -> None:
        super().__init__()
        self.name2annot, ecterms, ecnames, counts = load_EC_annot(data_path)
        self.name2sequence = load_name2sequence(data_path)
        self.class_num = len(ecterms[task_type])

        names = load_name(data_path, split)
        self.data_path = data_path
        self.data = [[self.name2sequence[name], self.name2annot[name][task_type]] for name in names]

    def __getitem__(self, index) -> Tuple[str, np.ndarray]:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        i = 0
        while i < len(self.data):
            yield self.data[i]
            i += 1

