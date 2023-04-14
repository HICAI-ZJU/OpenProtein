import unittest
import os

from openprotein.utils import Accuracy, MeanSquaredError, Spearman, AveragePrecisionScore
import torch

class MetricsTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    def test_Accuracy(self):
        acc = Accuracy()
        true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.assertTrue(acc(true, pred), 1)

    def test_MeanSquaredError(self):
        mse = MeanSquaredError()
        true = [3, -0.5, 2, 7]
        pred = [2.5, 0.0, 2, 8]
        true = torch.Tensor(true)
        pred = torch.Tensor(pred)
        self.assertTrue(mse(true, pred), 0.375)

    def test_Spearman(self):
        spe = Spearman()
        true1 = [1, 2, 3, 4, 5]
        pred1 = [5, 6, 7, 8, 7]
        true1 = torch.Tensor(true1)
        pred1 = torch.Tensor(pred1)
        self.assertTrue(spe(true1, pred1), 0.8207826816681233)

    def test_AveragePrecisionScore(self):
        aps = AveragePrecisionScore()
        true1 = [1, 1, 0, 1]
        pred1 = [0.3, 0.4, 0.2, 0.1]
        true1 = torch.Tensor(true1)
        pred1 = torch.Tensor(pred1)
        self.assertTrue(aps(true1, pred1), 0.9166666666666665)


if __name__ == "__main__":
    unittest.main()
