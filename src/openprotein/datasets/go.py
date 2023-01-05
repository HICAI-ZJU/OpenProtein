from openprotein.data.factory import Data


class GO(Data):
    """
    GO: The goal of the Gene Ontology Consortium is to produce a dynamic, controlled vocabulary that can be applied
    to all eukaryotes even as knowledge of gene and protein roles in cells is accumulating and changing. The data are
    from https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data

     Args:
         path (str): path for the dataset

     Examples:
         1:

         >>> from openprotein.datasets import GO
         >>> data = GO("./resources/go/valid")
         >>> dataset = data.get_data()
         >>> print(dataset[0])
         MKWTNAGSRRGSKKAAPSARPLPVNLRLNDFSDDELHLATRRSTGNSPDAPPQAERVGYSQLTVLIAELRRSSRLGRSTCAEVTRHYPAIIYVFVFTRCLPQPNSCST

         2:

         >>> dl = data.get_dataloader(batch_size=2)
         >>> for x, y in dl
         >>>    print(x)
         ('MSISDTVKRAREAFNSGKTRSLQFRIQQLEALQRMINENLKSISGALASDLGKNEWTSYYEEVAHVLEELDTTIKELPDWAEDEPVAKTRQTQQDDLYIHSEPLGVVLVIGAWNYPFNLTIQPMVGAVAAGNAVILKPSEVSGHMADLLATLIPQYMDQNLYLVVKGGVPETTELLKERFDHIMYTGSTAVGKIVMAAAAKHLTPVTLELGGKSPCYVDKDCDLDVACRRIAWGKFMNSGQTCVAPDYILCDPSIQNQIVEKLKKSLKDFYGEDAKQSRDYGRIINDRHFQRVKGLIDNQKVAHGGTWDQSSRYIAPTILVDVDPQSPVMQEEIFGPVMPIVCVRSLEEAIQFINQREKPLALYVFSNNEKVIKKMIAETSSGGVTANDVIVHITVPTLPFGGVGNSGMGAYHGKKSFETFSHRRSCLVKSLLNEEAHKARYPPSPAKMPRH', 'IIGGHEAKPHSRPYMAYLQIMDEYSGSKKCGGFLIREDFVLTAAHCSGSKIQVTLGAHNIKEQEKMQQIIPVVKIIPHPAYNSKTISNDIMLLKLKSKAKRSSAVKPLNLPRRNVKVKPGDVCYVAGWGKLGPMGKYSDTLQEVELTVQEDQKCESYLKNYFDKANEICAGDPKIKRASFRGDSGGPLVCKKVAAGIVSYGQNDGSTPRAFTKVSTFLSWIKKTMKKS')

     """

    def __init__(self, path: str):
        super().__init__(path)