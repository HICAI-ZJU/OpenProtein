import argparse
from openprotein import Esm1b, Esm1bConfig
from openprotein.data import MaskedConverter, Alphabet


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("-s", '--seq', type=str, default='RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE',\
                    help='protein seqences')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()

    converter = MaskedConverter.build_convert()
    alphabet = Alphabet.build_alphabet()

    esm_conf = Esm1bConfig()

    origin_tokens, masked_tokens, target_tokens = converter(args.seq)
    model = Esm1b(esm_conf, alphabet)

    feature = model(masked_tokens)
    print(f"The origin sequence is: {args.seq},\nAfter encoding, its feature is {feature}")