from transformers import BertModel, BertTokenizer


class ProteinBert(object):

    @staticmethod
    def get_proteinBert():
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        return tokenizer, model