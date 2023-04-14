import argparse
from openprotein import ProteinBert
import re


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("-s", '--seq', type=str, default='A E T C Z A O',\
                    help='protein seqences')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()

    tokenizer, model = ProteinBert.get_proteinBert()

    sequence_Example = re.sub(r"[UZOB]", "X", args.seq)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    feature = model(**encoded_input)

    print(f"The origin sequence is: {args.seq},\nAfter encoding, its feature is {feature}")