from transformers import BertTokenizer, BertModel

a = BertTokenizer.from_pretrained('bert-base-uncased')
b = BertModel.from_pretrained('bert-base-uncased')
