from typing import Tuple
import os
import pandas as pd
import re
import numpy as np


def load_flip_dataset(path, file_name):
    """returns dataframe of train, (val), test sets, with max_length param"""
    data_dir = os.path.join(path, file_name)
    df = pd.read_csv(data_dir)
    df.sequence = df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper()))  # remove special characters

    test = df[df.set == 'test']
    train = df[(df.set == 'train')]
    return train, test


class FLIPDataset:
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index) -> Tuple[str, np.ndarray]:
        item = self.data.iloc[index]
        if len(item['sequence']) >= 1000:
            return item['sequence'][:1000], item['target']
        else:
            return item['sequence'], item['target']

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        i = 0
        while i < len(self.data):
            yield self[i]
            i += 1
