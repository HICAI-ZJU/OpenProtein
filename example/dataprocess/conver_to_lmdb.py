import os
import pickle

import lmdb
import argparse
from ec import ECFunctionDataset
from flip import load_flip_dataset, FLIPDataset
from go import GOFunctionDataset


def parser_arguments():
    parser = argparse.ArgumentParser(description='Input Some Parameters')
    parser.add_argument("data", type=str, default=None,\
                    help='dataset name')
    parser.add_argument("-p", '--path', type=str, default='./',\
                    help='original dataset path')
    parser.add_argument("-o", '--output', type=str, default='./',\
                    help='converted dataset path')
    return parser.parse_args()


def convert_to_lmdb(data, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    txn = txn
    for i, data in enumerate(data):
        txn.put(str(i).encode(), pickle.dumps(data))
    txn.put("data_size".encode(), str(len(data)).encode())
    txn.commit()
    env.close()


def store_ec(path, output_path):
    ec_data_split = ["train", "test"]
    for split in ec_data_split:
        data = ECFunctionDataset(data_path=path, task_type='ec', split=split)
        lmdb_path = os.path.join(output_path, split)
        convert_to_lmdb(data, lmdb_path)
    print(f"Successfully stored ec dataset")


def store_flip(path, output_path):
    files = os.listdir(path)
    for file in files:
        if "csv" not in file:
            continue
        train, test = load_flip_dataset("../../test/resources/flip", file)
        train = FLIPDataset(train)
        name = os.path.splitext(file)[0]
        convert_to_lmdb(train, os.path.join(output_path, name+"_train"))
        test = FLIPDataset(test)
        convert_to_lmdb(test, os.path.join(output_path, name+"_test"))
    print(f"Successfully stored flip dataset")


def store_go(path, output_path):
    task_types = ['mf', 'bp', 'cc']
    for type in task_types:
        train_data = GOFunctionDataset(data_path=path, task_type=type, split="train")
        lmdb_path = os.path.join(output_path, f"{type}_train")
        convert_to_lmdb(train_data, lmdb_path)

        test_data = GOFunctionDataset(data_path=path, task_type=type, split="test")
        lmdb_path = os.path.join(output_path, f"{type}_test")
        convert_to_lmdb(test_data, lmdb_path)

    print(f"Successfully stored go dataset")


def store_factory(args):
    return globals()[f"store_{args.data}"]


if __name__ == "__main__":
    args = parser_arguments()
    store = store_factory(args)
    store(args.path, args.output)
