import unittest
import os

from torch.utils.data import DataLoader

from openprotein.task import ec
from openprotein.task import flip
from openprotein.models import esm1b


class EcTest(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.path = r".\resources\data\ec"
        self.task_type = 'ec'
        self.Args = flip.Args()
        self.args = self.Args.args_parser()
        self.alphabet = esm1b.Alphabet.build_alphabet()
        self.batch_str = (
        'GSAAGPPLSEDDKLQGAASHVPEGFDPTGPAGLGRPTPGLSQGPGKETLESALIALDSEKPKKLRFHPKQLYFSARQGELQKVLLMLVDGIDPNFKMEHQNKRSPLHAAAEAGHVDICHMLVQAGANIDTCSEDQRTPLMEAAENNHLEAVKYLIKAGALVDPKDAEGSTCLHLAAKKGHYEVVQYLLSNGQMDVNCQDDGGWTPMIWATEYKHVDLVKLLLSKGSDINIRDNEENICLHWAAFSGCVDIAEILLAAKCDLHAVNIHGDSPLHIAARENRYDCVVLFLSRDSDVTLKNKEGETPLQCASLNSQVWSALQMSKALQDSA',
        'MALLDVCGAPRGQRPESALPVAGSGRRSDPGHYSFSMRSPELALPRGMQPTEFFQSLGGDGERNVQIEMAHGTTTLAFKFQHGVIAAVDSRASAGSYISALRVNKVIEINPYLLGTMSGCAADCQYWERLLAKECRLYYLRNGERISVSAASKLLSNMMCQYRGMGLSMGSMICGWDKKGPGLYYVDEHGTRLSGNMFSTGSGNTYAYGVMDSGYRPNLSPEEAYDLGRRAIAYATHRDSYSGGVVNMYHMKEDGWVKVESTDVSDLLHQYREANQ')
        self.class_num = 538

    def test_ProteinFunctionDecoder(self):
        self.model = esm1b.ProteinBertModel(self.args, self.alphabet)
        self.model = self.model.load(self.args, self.alphabet).eval()
        self.converter = flip.Convert(self.alphabet)
        batch_tokens = self.converter(self.batch_str)
        logits = self.model(batch_tokens)['logits']
        results = ec.ProteinFunctionDecoder(self.args.embed_dim, self.class_num)(logits, self.batch_str)
        print(results)


if __name__ == "__main__":
    unittest.main()
