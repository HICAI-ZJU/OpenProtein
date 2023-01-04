import unittest
import os

import argparse

from torch import optim
from torch.optim.lr_scheduler import StepLR
from openprotein.utils import Accuracy

from openprotein.data import MaskedConverter, Alphabet
from openprotein.datasets import Uniref
from openprotein.models import Esm1b


class Esm1bTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        proteinseq_toks = {
            'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
                     'X', 'B', 'U', 'Z', 'O', '.', '-']
        }
        self.converter = MaskedConverter.build_convert(proteinseq_toks)
        self.alphabet = Alphabet.build_alphabet(proteinseq_toks)

        self.args = {'num_layers': 33, 'embed_dim': 1280, 'logit_bias': True, 'ffn_embed_dim': 5120, 'attention_heads': 20,
                'max_positions': 1024, 'emb_layer_norm_before': True, 'checkpoint_path': 'E:/esm1b_t33_650M_UR50S.pt'}
        self.args = argparse.Namespace(**self.args)
        self.model = Esm1b(self.args, self.alphabet)

    def test_load(self):
        self.model = self.model.load(self.args,self.alphabet)
        print(self.model)

    def test_forward(self):
        self.data = Uniref("./resources/uniref50/valid")
        f = lambda x: self.converter(x)
        dl = self.data.get_dataloader(collate_fn=f)

        for origin_tokens, masked_tokens, target_tokens in dl:
            result = self.model(masked_tokens)['logits']
            print(result)
            break


if __name__ == "__main__":
    unittest.main()