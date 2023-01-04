import unittest
import os

from openprotein.task import flip
from openprotein.models import esm1b


class FilpTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.path = r"../src/resources/splits/one_vs_rest.csv"
        self.Args = flip.Args()
        self.args = self.Args.args_parser()
        self.alphabet = esm1b.Alphabet.build_alphabet()
        self.batch_str = ('MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTFTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH',)

    def test_convert(self):
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        print(batch_tokens)

    def test_SequenceRegressionDecoder(self):
        self.model = esm1b.ProteinBertModel(self.args, self.alphabet)
        self.model = self.model.load(self.args, self.alphabet).eval()
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        logits = self.model(batch_tokens)['logits']
        results = flip.SequenceRegressionDecoder(self.args.embed_dim)(logits, self.batch_str)
        print(results)


if __name__ == "__main__":
    unittest.main()
