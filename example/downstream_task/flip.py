import argparse
from openprotein import Esm1b, Esm1bConfig
from openprotein.data import Alphabet, TaskConvert
from openprotein.task import SequenceRegressionDecoder


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("-s", '--seq', type=str, default='RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE',\
                    help='protein seqences')
    parser.add_argument("-p", '--path', type=str, default=None,\
                    help='pre-training model parameter')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()

    alphabet = Alphabet.build_alphabet()
    esm_conf = Esm1bConfig(checkpoint_path=args.path)

    model = Esm1b.load(esm_conf, alphabet).eval()

    converter = TaskConvert(alphabet)
    batch_tokens = converter(args.seq)
    
    feature = model(batch_tokens)

    srd = SequenceRegressionDecoder(esm_conf.embed_dim)
    output = srd(feature, args.seq)
    print(output)