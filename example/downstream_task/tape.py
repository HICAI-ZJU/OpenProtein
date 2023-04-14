import argparse
from openprotein import Esm1b, Esm1bConfig
from openprotein.data import Alphabet, TaskConvert
from openprotein.task import SequenceRegressionDecoder, SequenceToSequenceClassificaitonDecoder, SequenceClassificaitonDecoder, ProteinContactMapDecoder


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("-s", '--seq', type=str, default='RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE',\
                    help='protein seqences')
    parser.add_argument("-p", '--path', type=str, default=None,\
                    help='pre-training model parameter')
    parser.add_argument("-t", '--task', type=str, default=None,\
                    help="downstream task, such as 'contact_map', 'remote_homology', 'secondary_structure', 'fluorescence'")
    return parser.parse_args()


def fluorescence(args, logits):
    srd = SequenceRegressionDecoder(args.embed_dim)
    feature = srd(logits, args.seq)
    print(feature)


def secondary_structure(args, logits, label_num=1):
    stscd = SequenceToSequenceClassificaitonDecoder(args.embed_dim, label_num)
    feature = stscd(logits, args.seq)
    print(feature)


def remote_homology(args, logits, label_num=1):
    scd = SequenceClassificaitonDecoder(model.args.embed_dim, label_num)
    feature = scd(logits, args.seq)
    print(feature)


def contact_map(args, logits):
    pcmd = ProteinContactMapDecoder(args.embed_dim)
    feature = pcmd(logits, args.seq)
    print(feature)


def load_task(task):
    assert task in ['contact_map', 'remote_homology', 'secondary_structure', 'fluorescence'], \
        "task should be one of 'remote_homology', 'secondary_structure', 'fluorescence'"
    return globals()[task]


if __name__ == "__main__":
    args = parser_arguments()
    task = load_task(args.task)

    alphabet = Alphabet.build_alphabet()
    esm_conf = Esm1bConfig(checkpoint_path=args.path)

    model = Esm1b.load(esm_conf, alphabet).eval()

    converter = TaskConvert(alphabet)
    batch_tokens = converter(args.seq)
    
    feature = model(batch_tokens)
    output = task(feature)
    print(output)
