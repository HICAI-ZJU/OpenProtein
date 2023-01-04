from typing import *

import logging
import lmdb
import pickle as pkl

from openprotein.core import Components
from openprotein.utils import convert_to_bytes, convert_to_str


class Data(metaclass=Components):
    """
    A basic interface for data.

    The methods `get_data` and `get_dataloader` are defined. All data subclasses need to implement this interface.

    Args:
        path (str): path for the dataset.

    """

    def __init__(self, path):
        self._data = DataFactory.load(self, path)

    def get_data(self):
        """
        Get the dataset instance of the current dataset.

        Args:
            No Args.

        Returns:
            torch.utils.data.Dataset
        """
        return self._data.get_data()

    def get_dataloader(self, batch_size: int = None, shuffle: bool = False, sampler=None,
                       batch_sampler=None,
                       num_workers: int = 0, collate_fn=None,
                       pin_memory: bool = False, drop_last: bool = False):
        """
        Get the dataloader of the current dataset.
        For details, see: `https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`

        Args:
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            sampler (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.
            batch_sampler (Sampler or Iterable, optional): like sampler, but returns a batch of indices at a time.
            num_workers (int, optional): how many subprocesses to use for data loading. (default: 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory (bool, optional):
                If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            drop_last (bool, optional):
                set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
                (default: False)

        Returns:
            torch.utils.data.DataLoader
        """
        return self._data.get_dataloader(batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last)

    def __str__(self) -> str:
        return self.__name__


class DataFactory(object):
    """
    Static Data Factory. Load different implementations of the dataset depending on the current runtime environment.
    """

    @staticmethod
    def load(instance: Data, path: str) -> Data:
        """
        Load different implementations of the dataset depending on the current runtime environment.

        Args:
            instance (Data): instance for Data class
            path (str): path for the dataset file.

        Returns:
            A instance of implementing the Data interface

        Examples:
            >>> data = DataFactory.load("./uniref50/valid")

        """
        if instance._backend == "pt":
            return PTDataFactory(path)
        elif instance._backend == "ms":
            raise NotImplemented


class PTDataFactory(Data):
    """
    Factory for generating PyTorch datasets

    Args:
        path (str): path for the dataset file.

    Raises:
        ImportError: No module named torch.
    """

    try:
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        logging.error("No module named torch.")
        raise ImportError("No module named torch.") from e

    def __init__(self, path: str):
        self._dataset = self.PTDataset(path)

    def get_data(self) -> Dataset:
        """
        Get the PyTorch implementation of the current dataset

        Args:
            No args.

        Returns:
             torch.utils.data.Dataset
        """
        return self._dataset

    def get_dataloader(self, batch_size: int = None, shuffle: bool = False, sampler=None,
                       batch_sampler=None,
                       num_workers: int = 0, collate_fn=None,
                       pin_memory: bool = False, drop_last: bool = False) -> DataLoader:
        """
        Get the PyTorch implementation of Dataloader for the current dataset
        For details, see: `https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`

        Args:
            batch_size (int, optional): how many samples per batch to load (default: 1).
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch (default: False).
            sampler (Sampler or Iterable, optional): defines the strategy to draw samples from the dataset.
            batch_sampler (Sampler or Iterable, optional): like sampler, but returns a batch of indices at a time.
            num_workers (int, optional): how many subprocesses to use for data loading. (default: 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
            pin_memory (bool, optional):
                If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.
            drop_last (bool, optional): (default: False)
                set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
                If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

        Returns:
            torch.utils.data.DataLoader
        """
        return self.DataLoader(self._dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                               num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                               drop_last=drop_last)

    class PTDataset(Dataset):
        """
        Implementing the PyTorch dataset

        Args:
            lmdb_path (str): path for the lmdb dataset
        """

        def __init__(self, lmdb_path: str):
            self._load_lmdb(lmdb_path)
            data_size = self._cur.get("data_size".encode()).decode() if self._cur.get("data_size".encode()) \
                else pkl.loads(self._cur.get(b'num_examples'))
            self._data_size = int(data_size)

        def _load_lmdb(self, lmdb_path: str):
            """
            Load the dataset from lmdb path

            Args:
                lmdb_path (str): path for the lmdb dataset

            Raises:
                FileNotFoundError: there is no available file.
            """
            try:
                self._data = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                self._cur = self._data.begin(write=False).cursor()
                logging.info(f"load {self.__class__} sucessfully")
            except Exception as e:
                logging.warning(e)
                raise FileNotFoundError(e) from e

        def __len__(self):
            return self._data_size

        def __getitem__(self, index: Union[str, int, slice, list]) -> Union[str, List[str]]:
            if isinstance(index, slice):
                start = index.start if index.start else 0
                stop = index.stop if index.stop else self._data_size
                step = index.step if index.step else 1
                index = list(range(start, stop, step))
                return self._get_multi_data(index)
            elif isinstance(index, list):
                return self._get_multi_data(index)
            else:
                return convert_to_str(self._cur.get(str(index).encode()))

        def _get_multi_data(self, index: List[int]) -> List[str]:
            """
            Get protein sequence list by index

            Args:
                index (List[int]): index used to fetch the value

            Returns:
                A list consisting of protein sequences
            """
            index = convert_to_bytes(index)
            result = convert_to_str(self._cur.getmulti(index))
            return self._extract_protein_sequence(result)

        @staticmethod
        def _extract_protein_sequence(result: List[List[str]]) -> List[str]:
            """
            Extract protein sequences from (indexed, protein) sequences

            Args:
                result (List[List[str]]): List[indexed, protein] sequences

            Returns:
                A list consisting of protein sequences
            """
            return list(zip(*result))[1]
