import argparse
from openprotein import GearNet, GearNetConfig


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("-s", '--seq', type=str, default='RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE',\
                    help='protein seqences')
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()

    gearNet_args = GearNetConfig()

    model = GearNet.load_args(gearNet_args)

    feature = model(args.seq)
    print(f"The origin sequence is: {args.seq},\nAfter encoding, its feature is {feature}")