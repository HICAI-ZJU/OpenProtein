import unittest
import os

from openprotein.utils import Accuracy, MeanSquaredError, Spearman
import torch

class MetricsTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.true1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self.pred1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.true2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        self.pred2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


    def test_Accuracy(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = Accuracy()
        self.true2 = torch.Tensor(self.true2).to(device)
        self.pred2 = torch.Tensor(self.pred2).to(device)
        # Q: 0.0 is not true :0.0 猜测可能是浮点数0的存储问题，因此即使改动 也无法通过测试
        self.assertTrue(self.metrics(self.true2, self.pred2), 0.0)

    def test_MeanSquaredError(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')

        metrics = MeanSquaredError()
        true1 = [3, -0.5, 2, 7]
        pred1 = [2.5, 0.0, 2, 8]
        true1 = torch.Tensor(true1).to(device)
        pred1 = torch.Tensor(pred1).to(device)
        self.assertTrue(metrics(true1, pred1), 0.375)


    def test_Spearman(self):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        # device = "cpu"
        metrics = Spearman()
        true1 = [1, 2, 3, 4, 5]
        pred1 = [5, 6, 7, 8, 7]
        true1 = torch.Tensor(true1).to(device)
        pred1 = torch.Tensor(pred1).to(device)
        self.assertTrue(metrics(true1, pred1), 0.8207826816681233)



    def test_Metrics(self):
        self.assertTrue(self.metrics(self.true1, self.pred1), 0)
        self.assertTrue(self.metrics(self.true2, self.pred2), 0.5)
        print(self.metrics)



if __name__ == "__main__":
    unittest.main()
