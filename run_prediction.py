from src.data.preprocess_dataset import create_dataloader
from src.models.model import HGNModel, Validation
from src.models.document_retrieval import DocumentRetrieval
import torch
import json
import os


data_path = '/workspace/ml-workspace/thesis_git/HSGN/data/'
model_path = '/workspace/ml-workspace/thesis_git/HSGN/models'

print("Preprocessing data")
hotpotqa_path = 'external/'
print("Loading HotpotQA")
with open(os.path.join(data_path, hotpotqa_path, "input.json"), "r") as f:
    hotpot = json.load(f)
hotpot = hotpot[0:2]

device = 'cuda'
doc_retr_model_path = '/workspace/ml-workspace/thesis_git/HSGN/models/doc_retrieval'
doc_retr = DocumentRetrieval(device, doc_retr_model_path)
doc_retr.predict_relevant_docs(hotpot)


output = create_dataloader(hotpot)
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

print("Staring evaluation")
validation = Validation(model, hotpot, list_graphs,
                        tensor_input_ids, tensor_attention_masks,
                        tensor_token_type_ids,
                        list_span_idx)
metrics = validation.do_validation()
print(metrics)
print("Done")
