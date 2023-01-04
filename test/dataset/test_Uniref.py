import unittest
import os

from openprotein.data import MaskedConverter
from openprotein.datasets import Uniref


class UnirefTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        # self.args = DataConfig()
        # self.args.path = "./resources/uniref50/valid"
        path = "./resources/uniref50/valid"
        self.data = Uniref(path)

    def test_get_data(self):
        data = self.data.get_data()
        x = ['0', "1", "2", "3", "4"]
        result = data[x]
        self.assertEqual(data[:4:2], result[:4:2])

    def test_get_dataloader(self):
        proteinseq_toks = {
            'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                     'X', 'B', 'U', 'Z', 'O', '.', '-']
        }
        converter = MaskedConverter(proteinseq_toks["toks"])
        f = lambda x: converter(x)
        dl = self.data.get_dataloader(batch_size=4, collate_fn=f)
        for i, j, k in dl:
            print(i, j)


if __name__ == "__main__":
    unittest.main()