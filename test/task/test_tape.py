import unittest
import os

from openprotein.task import tape
from openprotein.task import flip
from openprotein.models import esm1b


class TapeTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.path = r".\resources\data"
        self.batch_size = 4
        self.Args = flip.Args()
        self.args = self.Args.args_parser()
        self.alphabet = esm1b.Alphabet.build_alphabet()
        self.batch_str = ()
        self.class_num = 0

    def test_ProteinContactMapDecoder(self):
        self.batch_str = (
        'MHHHHHHSSGVDLGTENLYFQSNAMIERLLEIKKIRADRADKAVQRQEYRVANVAAELQKAERSVADYHVWRQEEEERRFAKAKQQTVLLKELETLRQEIALLREREAELKQRVAEVKVTLEQERTLLKQKQQEALQAHKTKEKFVQLQQQEIAEQSRQQQYQEELEQEEFRTVDII',)
        self.model = esm1b.ProteinBertModel(self.args, self.alphabet)
        self.model = self.model.load(self.args, self.alphabet).eval()
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        logits = self.model(batch_tokens)['logits']
        results = tape.ProteinContactMapDecoder(self.args.embed_dim)(logits)
        print(results)

    def test_SequenceToSequenceClassificaitonDecoder(self):
        self.class_num = 3
        self.batch_str = (
        'NTIQQLMMILNSASDQPSENLISYFNNCTVNPKESILKRVKDIGYIFKEKFAKAVGQGCVEIGSQRYKLGVRLYYRVMESMLKSEEERLSIQNFSKLLNDNIFHMSLLACALEVVMATYSRSTSQNLDSGTDLSFPWILNVLNLKAFDFYKVIESFIKAEGNLTREMIKHLERCEHRIMESLAWLSDSPLFDLIKQSKDREGKSTSLSLFYKKVYRLAYLRLNTLCERLLSEHPELEHIIWTLFQHTLQNEYELMRDRHLDQIMMCSMYGICKVKNIDLKFKIIVTAYKDLPHAVQETFKRVLIKEEEYDSIIVFYNSVFMQRLKTNILQYASTRPPTLSPIPHIPR',)
        self.model = esm1b.ProteinBertModel(self.args, self.alphabet)
        self.model = self.model.load(self.args, self.alphabet).eval()
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        logits = self.model(batch_tokens)['logits']
        results = tape.SequenceToSequenceClassificaitonDecoder(self.args.embed_dim, self.class_num)(logits)
        print(results)

    def test_SequenceClassificaitonDecoder(self):
        self.batch_str = (
        'PRGNKVAITNAGGPGVLTADELDKRGLKLATLEEKTIEELRSFLPPAAVKNPVDIASARGEDYYRTAKLLLQDPNVDLIAICVVPTFAGTLTEHAEGIIRAVKEVNNEKPVLAFAGYVSEKAKELLEKNGIPTYERPEDVASAAYALVEQAKNVGI',)
        self.class_num = 1193
        self.model = esm1b.ProteinBertModel(self.args, self.alphabet)
        self.model = self.model.load(self.args, self.alphabet).eval()
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        logits = self.model(batch_tokens)['logits']
        results = tape.SequenceClassificaitonDecoder(self.args.embed_dim, self.class_num)(logits, self.batch_str)
        print(results)


if __name__ == "__main__":
    unittest.main()
