import unittest
import os

from openprotein.data import MaskedConverter
from openprotein.datasets import Tape


class TapeTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        path = "./resources/tape/fluorescence_test.lmdb"
        self.data = Tape(path)

    def test_get_data(self):
        data = self.data.get_data()
        x = ['0', "1", "2", "3", "4"]
        result = data[x]
        self.assertEqual(data[:4:2][1]["primary"], result[:4:2][1]["primary"])

    def test_get_dataloader(self):
        dl = self.data.get_dataloader(batch_size=2)
        for x in dl:
            print(x)
            break


if __name__ == "__main__":
    unittest.main()