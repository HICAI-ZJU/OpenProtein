import unittest
import os

from openprotein.data import MaskedConverter
from openprotein.datasets import Uniref


class UnirefTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        path = "./resources/uniref50/valid"
        self.data = Uniref(path)

    def test_get_data(self):
        data = self.data.get_data()
        x = ['0', "1", "2", "3", "4"]
        result = data[x]
        self.assertEqual(data[:4:2], result[:4:2])

    def test_get_dataloader(self):
        converter = MaskedConverter.build_convert()
        f = lambda x: converter(x)
        dl = self.data.get_dataloader(batch_size=4, collate_fn=f)
        for i, j, k in dl:
            print(i, j)
            break


if __name__ == "__main__":
    unittest.main()