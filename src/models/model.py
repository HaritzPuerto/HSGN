from transformers import AdamW, BertConfig
import os
import json
import pickle
import warnings
import re
import math

from os import listdir
from os.path import isfile, join
from dgl.data.utils import load_graphs

from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

import dgl
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

warnings.filterwarnings(action='once')

os.environ['DGLBACKEND'] = 'pytorch'

import random
random_seed = 2020
# Set the seed value all over the place to make this reproducible.
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

pretrained_weights = 'bert-base-cased'
#pretrained_weights = 'bert-large-cased-whole-word-masking'
device = 'cuda'

weights = torch.tensor([1., 30.9, 31.], device=device)

# %%
loss_fn_ans_type = nn.CrossEntropyLoss(weights)

# %%
def get_sent_node_from_srl_node(graph, srl_node, list_srl_nodes):
    _, out_srl = graph.out_edges(srl_node)
    list_sent = list(set(out_srl.numpy()) - set(list_srl_nodes))
    # there is only one element by construction of the graph
    return list_sent[0]


# %%
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# %%
loss_fn = LabelSmoothingLoss()

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# %%
bert_dim = 768 # default
if 'large' in pretrained_weights:
    bert_dim = 1024
if 'albert-xxlarge-v2' == pretrained_weights:
    bert_dim = 4096
dict_params = {'in_feats': bert_dim, 'out_feats': bert_dim, 'feat_drop': 0.2, 'attn_drop': 0.1, 'hidden_size_classifier': bert_dim,
               'weight_sent_loss': 1, 'weight_srl_loss': 1, 'weight_ent_loss': 1, 'bi_gru_layers': 1,
               'weight_span_loss': 2, 'weight_ans_type_loss': 1, 'span_drop': 0.2,
               'gat_layers': 2, 'accumulation_steps': 1, 'residual': True,}
class HGNModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if pretrained_weights == 'albert-xxlarge-v2':
            self.bert = AlbertModel(config)
        else:
            self.bert = BertModel(config)
        
        # span prediction
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), GeLU(), nn.Dropout(dict_params['span_drop']),
                                        nn.Linear(config.hidden_size, 2))
        # init weights
        self.init_weights()
        # params
        
    
    def forward(
        self,
        graph=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        train=True
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        
        span_loss, start_logits, end_logits = self.span_prediction(sequence_output, start_positions, end_positions)
        return {'span': {'loss': span_loss, 'start_logits': start_logits, 'end_logits': end_logits}}  
    
    def span_prediction(self, sequence_output, start_positions, end_positions):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if ((start_positions is not None and end_positions is not None) and
            (start_positions != -1 and end_positions != -1)):
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        return (total_loss,  start_logits, end_logits)

class Validation():

    def __init__(self, model, dataset, validation_dataloader,
                 tensor_input_ids, tensor_attention_masks, tensor_token_type_ids):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.validation_dataloader = validation_dataloader
        self.tensor_input_ids = tensor_input_ids
        self.tensor_attention_masks = tensor_attention_masks
        self.tensor_token_type_ids = tensor_token_type_ids
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights, 
                                                       do_basic_tokenize=False, clean_text=False)

    def get_answer_predictions(self, dict_ins2dict_doc2pred):
        output_pred_sp = {}
        output_predictions_ans = {}
        for step, b_graph in enumerate(tqdm(self.validation_dataloader)): 
            with torch.no_grad():
                output = self.model(b_graph,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0).to(device),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0).to(device),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0).to(device), 
                               train=False)
            #answer
            predicted_ans = ""
            predicted_ans = self.__get_pred_ans_str(self.tensor_input_ids[step], output)
            _id = self.dataset[step]['_id']
            output_predictions_ans[_id] = predicted_ans
            
        return {'answer': output_predictions_ans, 'sp': output_pred_sp}

    def __get_pred_ans_str(self, input_ids, output):
        st, end = self.__get_st_end_span_idx(output['span']['start_logits'].squeeze(),
                                             output['span']['end_logits'].squeeze())
        return self.__get_str_span(input_ids, st, end)

    def __get_str_span(self, input_ids, st, end):
        return self.tokenizer.decode(input_ids[st:end])
    
    def __get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes
    
    def __get_st_end_span_idx(self, start_logits, end_logits, max_answer_length = 30):
        start_indexes = self.__get_best_indexes(start_logits, 10)
        end_indexes = self.__get_best_indexes(end_logits, 10)
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= 512:
                    continue
                if end_index >= 512:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                return (start_index, end_index)
        return (0, 0)