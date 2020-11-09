from src.data.preprocess_dataset import create_dataloader
from src.models.model import HGNModel, Validation
from src.models.document_retrieval import DocumentRetrieval
import torch
import json
import os
import sys, subprocess

def convert_sae_doc_ret_output2our_input(hotpot):
    with open('SAE/output/pred_gold_idx.json', 'r') as f:
        pred_gold_idx = json.load(f)
    out = dict()
    for ins_idx, pred in enumerate(pred_gold_idx[:10]):
        dict_doc2pred = dict()
        if len(hotpot[ins_idx]['context']) == 2:
            dict_doc2pred = {0: 1, 1: 1}
        else:
            for i in range(len(hotpot[ins_idx]['context'])):
                if i in pred:
                    dict_doc2pred[i] = 1
                else:
                    dict_doc2pred[i] = 0
        out[ins_idx] = dict_doc2pred
    return out


device = 'cuda'
data_path = 'data/'
pretrained_weights = 'bert-large-cased-whole-word-masking'
#pretrained_weights = 'bert-base-cased'
model_path = 'models/graph_model'
doc_retr_model_path = 'models/doc_retrieval'

print("Preprocessing data")
print("Loading HotpotQA")
hotpotqa_path = 'external/'
input_file = os.path.join(data_path, hotpotqa_path, "input.json")
with open(input_file, "r") as f:
    hotpot = json.load(f)
hotpot = hotpot[:10]
with open("input_sample.json", 'w+') as f:
    json.dump(hotpot, f)

dict_ins2dict_doc2pred = convert_sae_doc_ret_output2our_input(hotpot)
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
