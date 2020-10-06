from src.data.preprocess_dataset import create_dataloader
from src.models.model import HGNModel, Validation
from src.models.document_retrieval import DocumentRetrieval
import torch
import json
import os

device = 'cuda'
data_path = 'data/'
pretrained_weights = 'bert-large-cased-whole-word-masking'
#pretrained_weights = 'bert-base-cased'
model_path = 'models/graph_model'
doc_retr_model_path = 'models/doc_retrieval'

print("Preprocessing data")
print("Loading HotpotQA")
hotpotqa_path = 'external/'
with open(os.path.join(data_path, hotpotqa_path, "input.json"), "r") as f:
    hotpot = json.load(f)
print("Loading the document retrieval model")
doc_retr = DocumentRetrieval(device, doc_retr_model_path)
print("Computing the relevant documents")
dict_ins2dict_doc2pred = doc_retr.predict_relevant_docs(hotpot)
print("Creating graphs for the predicted relevant documents")
output = create_dataloader(hotpot, dict_ins2dict_doc2pred, pretrained_weights)
list_graphs = output['list_graphs']
list_context = output['list_context']
list_span_idx = output['list_span_idx']
print("Loading model")
model = HGNModel.from_pretrained(model_path)
model.cuda()

list_input_ids = [context['input_ids'] for context in list_context]
list_token_type_ids = [context['token_type_ids'] for context in list_context]
list_attention_masks = [context['attention_mask'] for context in list_context]

tensor_input_ids = torch.tensor(list_input_ids)
tensor_token_type_ids = torch.tensor(list_token_type_ids)
tensor_attention_masks = torch.tensor(list_attention_masks)

print("Computing answers")
validation = Validation(model, hotpot, list_graphs,
                        tensor_input_ids, tensor_attention_masks,
                        tensor_token_type_ids)
preds = validation.get_answer_predictions(dict_ins2dict_doc2pred)
with open('./pred.json', 'w+') as f:
    json.dump(preds, f)
print("Finished :)")
