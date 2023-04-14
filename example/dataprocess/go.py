from typing import Tuple
import os
import csv
import numpy as np


def load_GO_annot(filename):
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}

    with open(os.path.join(filename, 'nrPDB-GO_2019.06.18_annot.tsv'), mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts


def load_name(filename, split):
    name = []
    if split == 'valid':
        split = 'test'
    with open(os.path.join(filename, f'nrPDB-GO_2019.06.18_{split}.txt'), 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            name.append(row[0])
    return name


def load_name2sequence(filename):
    name2sequence = {}
    with open(os.path.join(filename, f'nrPDB-GO_2019.06.18_sequences.fasta'), 'r') as f:
        for line in f:
            if line.startswith('>'):
                name = line.replace('>', '').split()[0]
                name2sequence[name] = ''
            else:
                name2sequence[name] += line.replace('\n', '').strip()
    return name2sequence


class GOFunctionDataset:
    def __init__(self, data_path, task_type, split) -> None:
        super().__init__()
        self.name2annot, goterms, gonames, counts = load_GO_annot(data_path)
        self.name2sequence = load_name2sequence(data_path)
        self.class_num = len(goterms[task_type])

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