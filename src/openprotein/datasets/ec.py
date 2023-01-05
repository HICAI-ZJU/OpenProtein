from openprotein.data.factory import Data


class EC(Data):
    """
    EC: The ENZYME database in 2000 The ENZYME database is a repository of information related to the nomenclature of
    enzymes. The data are from https://github.com/flatironinstitute/DeepFRI/tree/master/preprocessing/data

     Args:
         path (str): path for the dataset

     Examples:
         1:

         >>> from openprotein.datasets import EC
         >>> data = EC(DATA_PATH)
         >>> dataset = data.get_data()
         >>> print(dataset[0])
         ['MAHHHHHHMALVSMRQLLDHAAENSYGLPAFNVNNLEQMRAIMEAADQVNAPVIVQASAGARKYAGAPFLRHLILAAVEEFPHIPVVMHQDHGASPDVCQRSIQLGFSSVMMDGSLLEDGKTPSSYEYNVNATRTVVNFSHACGVSVEGEIGVLGNLETGEAGEEDGVGAAGKLSHDQMLTSVEDAVRFVKDTGVDALAIAVGTSHGAYKFTRPPTGDVLRIDRIKEIHQALPNTHIVMHGSSSVPQEWLKVINEYGGNIGETYGVPVEEIVEGIKHGVRKVNIDTDLRLASTGAVRRYLAENPSDFDPRKYLGKTIEAMKQICLDRYLAFGCEGQAGKIKPVSLEKMASRYAKGELNQIVK', array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...

         2:

         >>> dl = data.get_dataloader(batch_size=2)
         >>> for x, y in dl
         >>>    print(x)
         ('MAHHHHHHMALVSMRQLLDHAAENSYGLPAFNVNNLEQMRAIMEAADQVNAPVIVQASAGARKYAGAPFLRHLILAAVEEFPHIPVVMHQDHGASPDVCQRSIQLGFSSVMMDGSLLEDGKTPSSYEYNVNATRTVVNFSHACGVSVEGEIGVLGNLETGEAGEEDGVGAAGKLSHDQMLTSVEDAVRFVKDTGVDALAIAVGTSHGAYKFTRPPTGDVLRIDRIKEIHQALPNTHIVMHGSSSVPQEWLKVINEYGGNIGETYGVPVEEIVEGIKHGVRKVNIDTDLRLASTGAVRRYLAENPSDFDPRKYLGKTIEAMKQICLDRYLAFGCEGQAGKIKPVSLEKMASRYAKGELNQIVK', 'MANRSHHNAGHRAMNALRKSGQKHSSESQLGSSEIGTTRHVYDVCDCLDTLAKLPDDSVQLIICDPPYNIMLADWDDHMDYIGWAKRWLAEAERVLSPTGSIAIFGGLQYQGEAGSGDLISIISHMRQNSKMLLANLIIWNYPNGMSAQRFFANRHEEIAWFAKTKKYFFDLDAVREPYDEETKAAYMKDKRLNPESVEKGRNPTNVWRMSRLNGNSLERVGHPTQKPAAVIERLVRALSHPGSTVLDFFAGSGVTARVAIQEGRNSICTDAAPVFKEYYQKQLTFLQDDGLIDKARSYEIVEGAANFGAALQRGDVAS')
     """

    def __init__(self, path: str):
        super().__init__(path)