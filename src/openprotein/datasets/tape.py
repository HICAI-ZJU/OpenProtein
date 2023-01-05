from openprotein.data.factory import Data


class Tape(Data):
    """
    The GB1 "four" variations set stems from https://elifesciences.org/articles/16965 in which mutations at four
    sites (V39, D40, G41 and V54) were probed against a binding assay. All splits are regression splits on the
    Fittness value reported in the https://elifesciences.org/articles/16965/figures#SD1-data. We use one_vs_rest
    which train is wild type and all single mutations, test is everything else.

     Args:
         path (str): path for the dataset

     Examples:
         1:
         >>> from openprotein.datasets import Tape
         >>> data = Tape("./resources/Tape/valid")
         >>> dataset = data.get_data()
         >>> print(dataset[0])
         MKWTNAGSRRGSKKAAPSARPLPVNLRLNDFSDDELHLATRRSTGNSPDAPPQAERVGYSQLTVLIAELRRSSRLGRSTCAEVTRHYPAIIYVFVFTRCLPQPNSCST

         2:

         >>> dl = data.get_dataloader(batch_size=2)
         >>> for x, y in dl
         >>>    print(x)
         {'primary': ['SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRDEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYDSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPVGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGTDEPYK', 'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRDEVKFEGDTLVNRIELKGIDFKEDGNILGHKLENNYNSLNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLGFVTAAGITHGMDELYK'], 'protein_length': tensor([237, 237]), 'log_fluorescence': tensor([[1.3010],
                [1.3012]]), 'num_mutations': tensor([5, 4])}
     """

    def __init__(self, path: str):
        super().__init__(path)