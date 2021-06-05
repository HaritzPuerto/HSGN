#!/usr/bin/env python
# coding: utf-8
# %%
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
from transformers import AlbertModel, AlbertTokenizer

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
torch.backends.cudnn.deterministic = True


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
data_path = "data/"
hotpot_qa_path = os.path.join(data_path, "external")

with open(os.path.join(hotpot_qa_path, "hotpot_train_v1.1.json"), "r") as f:
    hotpot_train = json.load(f)

with open(os.path.join(hotpot_qa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
    hotpot_dev = json.load(f)

# %%
set_easy = set()
set_med = set()
set_hard = set()
for ins_idx, ins in enumerate(hotpot_train):
    if ins['level'] == 'easy':
        set_easy.add(ins_idx)
    elif ins['level'] == 'medium':
        set_med.add(ins_idx)
    elif ins['level'] == 'hard':
        set_hard.add(ins_idx)

list_idx_curriculum_learning = []
for idx in set_easy:
    list_idx_curriculum_learning.append(idx)
for idx in set_med:
    list_idx_curriculum_learning.append(idx)
for idx in set_hard:
    list_idx_curriculum_learning.append(idx)


# %%
device = 'cuda'
pretrained_weights = 'bert-base-cased'
#pretrained_weights = 'bert-large-cased-whole-word-masking'
#pretrained_weights = 'albert-xxlarge-v2'
# ## HotpotQA Processing

# ## Processing

# %%
training_path = os.path.join(data_path, "processed/training/heterog_20201004_query_edges/")
dev_path = os.path.join(data_path, "processed/dev/heterog_20201004_query_edges/")

with open(os.path.join(training_path, 'list_span_idx.p'), 'rb') as f:
    list_span_idx = pickle.load(f)

tensor_input_ids = torch.load(os.path.join(training_path, 'tensor_input_ids.p'))
tensor_token_type_ids = torch.load(os.path.join(training_path, 'tensor_token_type_ids.p'))
tensor_attention_masks = torch.load(os.path.join(training_path, 'tensor_attention_masks.p'))

tensor_input_ids = tensor_input_ids.to(device)
tensor_attention_masks = tensor_attention_masks.to(device)
tensor_token_type_ids = tensor_token_type_ids.to(device)


# %%
with open(os.path.join(dev_path, 'list_span_idx.p'), 'rb') as f:
    dev_list_span_idx = pickle.load(f)
dev_tensor_input_ids = torch.load(os.path.join(dev_path, 'tensor_input_ids.p'))
dev_tensor_token_type_ids = torch.load(os.path.join(dev_path, 'tensor_token_type_ids.p'))
dev_tensor_attention_masks = torch.load(os.path.join(dev_path, 'tensor_attention_masks.p'))

dev_tensor_input_ids = dev_tensor_input_ids.to(device)
dev_tensor_attention_masks = dev_tensor_attention_masks.to(device)
dev_tensor_token_type_ids = dev_tensor_token_type_ids.to(device)


# %%
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# %%
def get_sent_node_from_srl_node(graph, srl_node, list_srl_nodes):
    _, out_srl = graph.out_edges(srl_node)
    list_sent = list(set(out_srl.numpy()) - set(list_srl_nodes))
    # there is only one element by construction of the graph
    return list_sent[0]

# %%
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


# %%


from transformers import AdamW, BertConfig

model = HGNModel.from_pretrained(
    pretrained_weights, # Use the 12-layer BERT model, with an cased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()


# # Optimizer & Learning Rate Scheduler
# 
# For the purposes of fine-tuning, the authors recommend choosing from the following values (from Appendix A.3 of the BERT paper):
# 
# * Learning rate (Adam): 5e-5, 3e-5, 2e-5
# * Number of epochs: 2, 3, 4
# 
# 
# 

# %%
# model.train()
# for step, b_graph in enumerate(tqdm(list_graphs)):
#     model.zero_grad()
#     # forward
#     input_ids=tensor_input_ids[step].unsqueeze(0).to(device)
#     attention_mask=tensor_attention_masks[step].unsqueeze(0).to(device)
#     token_type_ids=tensor_token_type_ids[step].unsqueeze(0).to(device) 
#     start_positions=torch.tensor([list_span_idx[step][0]], device='cuda')
#     end_positions=torch.tensor([list_span_idx[step][1]], device='cuda')
#     output = model(b_graph,
#                    input_ids=input_ids,
#                    attention_mask=attention_mask,
#                    token_type_ids=token_type_ids, 
#                    start_positions=start_positions,
#                    end_positions=end_positions)
#     total_loss = output['loss']
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     optimizer.step()
#     scheduler.step()
#     model.zero_grad()

# %%
lr = 1e-5
optimizer = AdamW(model.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5, 
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# %%


from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(hotpot_train) * epochs

#Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                             num_warmup_steps = 0, # Default value in run_glue.py
                                             num_training_steps = total_steps)

# scheduler_medium = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = len(train_dataloader_medium) * epochs)
# scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
#                                                                num_warmup_steps = 1000, # Default value in run_glue.py
#                                                                num_training_steps = total_steps,
#                                                                num_cycles=dict_params['num_cycles'])

# %%


train_batch_size = 1


# %%
import neptune
neptune.init(
    "haritz/srl-pred"
)
neptune.set_project('haritz/srl-pred')
PARAMS = {"num_epoch": epochs, 
          'lr': lr, 
          'pretrained_weights': pretrained_weights,
          'loss_fn': 'crossentropy_label_smoothing', 
          #'validation_size': len(validation_dataloader)*val_batch_size , 
          'random_seed': random_seed,
          'total_steps': total_steps, 
          'training_size': len(hotpot_train)*epochs, 
          'train_batch_size': train_batch_size,
          #'val_batch_size': val_batch_size, 
          'scheduler': 'get_linear_schedule_with_warmup'}
PARAMS.update(dict_params)


# # Training Loop

# %%
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# %%
def recall_at_k(prob_pos, k, labels):   
    k = min(k, len(prob_pos))
    _, idx_topk = torch.topk(prob_pos, k, dim=0)
    if sum(labels).item() == 0:
        if sum(labels[idx_topk]).item() == 0:
            return 1.0
        else:
            return 0.0
    return sum(labels[idx_topk]).item()/sum(labels).item()


# %%
def accuracy(pred, label):
    return torch.mean( (pred == label).type(torch.FloatTensor) ).item()


# # Evaluation Helpers

# %%
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def evaluation_metrics(prediction, truth):
    tp, fp, tn, fn = confusion(prediction, truth)
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1, prec, recall
    
def test_with_valid_tensors():
    prediction = torch.tensor([
        [1],
        [1.0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0]
    ])
    truth = torch.tensor([
        [1.0],
        [1],
        [0],
        [0],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    tp, fp, tn, fn = confusion(prediction, truth)
    
    assert tp == 2
    assert fp == 1
    assert tn == 3
    assert fn == 4
    assert evaluation_metrics(truth.view(-1), truth.view(-1)) == (1.0, 1.0, 1.0, 1.0)


# %%
import sys
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


# %%
if pretrained_weights == 'albert-xxlarge-v2':
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights, do_basic_tokenize=False, clean_text=False)
else:
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_basic_tokenize=False, clean_text=False)

# %%
import en_core_web_sm

wh_ans_len = {'which': 25, 'what':25, 'who':20, 'when':10, 'how':15, 'where':15, 'how many': 10, None: 15}
class Validation():

    def __init__(self, model, dataset, validation_dataloader, tokenizer,
                 tensor_input_ids, tensor_attention_masks, tensor_token_type_ids, list_span_idx):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.nlp = en_core_web_sm.load()
        self.validation_dataloader = validation_dataloader
        self.tokenizer = tokenizer
        self.tensor_input_ids = tensor_input_ids
        self.tensor_attention_masks = tensor_attention_masks
        self.tensor_token_type_ids = tensor_token_type_ids
        self.list_span_idx = list_span_idx
        
    def do_validation(self):       
        metrics = {'validation_loss': 0, 
                   'ans_em': 0, 'ans_f1': 0, 'ans_prec': 0 , 'ans_recall': 0,
                   'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
                   'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0
                   }
        # Evaluate data for one epoch       
        num_valid_examples = 0
        for step in trange(len(self.dataset)):
            num_valid_examples += 1
            with torch.no_grad(): 
                output = self.model(None,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0).to(device),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0).to(device),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0).to(device), 
                               start_positions=torch.tensor([self.list_span_idx[step][0]], device='cuda'),
                               end_positions=torch.tensor([self.list_span_idx[step][1]], device='cuda'),
                                    train=False)
                
            # Accumulate the validation loss.
            if output['span']['loss'] is not None:
                metrics['validation_loss'] += output['span']['loss'].item()
            # answer span prediction
            ## wh type
            query = self.dataset[step]['question']
            wh = self.__findWHword(query)
            max_ans_len = wh_ans_len[wh]
            golden_ans = self.dataset[step]['answer']
            predicted_ans = ""
            predicted_ans = self.__get_pred_ans_str(self.tensor_input_ids[step], output, max_ans_len)
            ans_em, ans_prec, ans_recall = self.update_answer_metrics(metrics, predicted_ans, golden_ans)
            # joint
            #self.update_joint_metrics(metrics, ans_em, ans_prec, ans_recall, sp_em, sp_prec, sp_recall)                
            
        #N = len(self.validation_dataloader)
        N = num_valid_examples
        for k in metrics.keys():
            metrics[k] /= N
        return metrics
    
    def get_answer_predictions(self, dict_ins2dict_doc2pred):
        output_pred_sp = {}
        output_predictions_ans = {}
        output_ent = {}
        output_srl = {}
        for step, b_graph in enumerate(tqdm(self.validation_dataloader)): 
            with torch.no_grad():
                output = self.model(b_graph,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0).to(device),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0).to(device),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0).to(device), 
                               train=False)
            _id = self.dataset[step]['_id']
            query = self.dataset[step]['question']
            wh = self.__findWHword(query)
            max_ans_len = wh_ans_len[wh]
            #answer
            predicted_ans = ""
            predicted_ans = self.__get_pred_ans_str(self.tensor_input_ids[step], output, max_ans_len)
            output_predictions_ans[_id] = predicted_ans
            # ent
            ent = self.__get_ent_str(b_graph, self.tensor_input_ids[step], output)
            output_ent[_id] = ent
            # srl
            srl = self.__get_srl_str(b_graph, self.tensor_input_ids[step], output)
            output_srl[_id] = srl
            #sp
#             prediction_sent = torch.argmax(output['sent']['probs'], dim=1)
#             sent_num = 0
#             dict_sent_num2str = dict()
#             for doc_idx, (doc_title, doc) in enumerate(self.dataset[step]['context']):
#                 if dict_ins2dict_doc2pred[step][doc_idx] == 1:
#                     for i, sent in enumerate(doc):
#                         dict_sent_num2str[sent_num] = {'sent': i, 'doc_title': doc_title}
#                         sent_num += 1
#             output_pred_sp[_id] = []
#             for i, pred in enumerate(prediction_sent):
#                 if pred == 1:
#                     output_pred_sp[_id].append([dict_sent_num2str[i]['doc_title'],
#                                                 dict_sent_num2str[i]['sent']])
        return {'answer': output_predictions_ans, 'sp': output_pred_sp,
                'ent': output_ent, 'srl': output_srl}
    
    def __get_pred_ans_str(self, input_ids, output, max_ans_len):
        st, end = self.__get_st_end_span_idx(output['span']['start_logits'].squeeze(),
                                             output['span']['end_logits'].squeeze(), max_ans_len)
        return self.__get_str_span(input_ids, st, end)
    
    def __get_ent_str(self, graph, input_ids, output):
        if 'ent' in graph.ntypes:
            ent_node = torch.argmax(output['ent']['probs'][:,1]).item()
            st, end = graph.nodes['ent'].data['st_end_idx'][ent_node]
            return self.__get_str_span(input_ids, st, end)
        else:
            return ""
    
    def __get_srl_str(self, graph, input_ids, output):
        if 'srl' in graph.ntypes:
            srl_node = torch.argmax(output['srl']['probs'][:,1]).item()
            st, end = graph.nodes['srl'].data['st_end_idx'][srl_node]
            return self.__get_str_span(input_ids, st, end)
        else: 
            return ""

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
        list_candidates = []
        list_scores = []
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
                list_scores.append(start_logits[start_index] + end_logits[end_index])
                list_candidates.append((start_index, end_index))
        if len(list_scores) == 0:
            return (0,0)
        else:
            best_span_idx = list_scores.index(max(list_scores))
            return list_candidates[best_span_idx]

    def __findWHword(self, sentence):
        candidate = ['when', 'how', 'where', 'which', 'what', 'who', 'how many']
        sentence = sentence.lower()
        doc = self.nlp(sentence)
        if 'how' in sentence.split() and 'how many' in sentence:
            return 'how many'
        for w in reversed(doc):
            if w.pos_ == 'NN': continue
            else:
                for can in candidate:
                    if can in w.text:
                        return can
                break
        whs = []
        for idx, token in enumerate(doc):
            for can in candidate:
                if can in token.text:
                    return can
        if 'name' in sentence.lower() or doc[-1].lemma_ == 'be' or doc[-1].pos_ == 'ADP':
            return 'what'
        return None
    
    def update_sp_metrics(self, metrics, prediction_sent, sent_labels):
        em, f1, prec, recall = evaluation_metrics(prediction_sent.type(torch.DoubleTensor), 
                                                  sent_labels.type(torch.DoubleTensor))
        metrics['sp_em'] += em
        metrics['sp_f1'] += f1
        metrics['sp_prec'] += prec
        metrics['sp_recall'] += recall
        return em, prec, recall
        
    def update_srl_metrics(self, metrics, prediction_srl, srl_labels, positive_probs):
        srl_eval = evaluation_metrics(prediction_srl.type(torch.DoubleTensor), 
                                      srl_labels.type(torch.DoubleTensor))
        metrics['srl_em'] += srl_eval[0]
        metrics['srl_f1'] += srl_eval[1]
        metrics['srl_prec'] += srl_eval[2]
        metrics['srl_recall'] += srl_eval[3]
        metrics['srl_recall@1'] += recall_at_k(positive_probs, 1, srl_labels)
        metrics['srl_recall@3'] += recall_at_k(positive_probs, 3, srl_labels)
        metrics['srl_recall@5'] += recall_at_k(positive_probs, 5, srl_labels)
        
    def update_ent_metrics(self, metrics, prediction_ent, ent_labels, positive_probs):
        ent_eval = evaluation_metrics(prediction_ent.type(torch.DoubleTensor), 
                                  ent_labels.type(torch.DoubleTensor))
        metrics['ent_em'] += ent_eval[0]
        metrics['ent_f1'] += ent_eval[1]
        metrics['ent_prec'] += ent_eval[2]
        metrics['ent_recall'] += ent_eval[3]
        metrics['ent_recall@1'] += recall_at_k(positive_probs, 1, ent_labels)
        metrics['ent_recall@3'] += recall_at_k(positive_probs, 3, ent_labels)
        metrics['ent_recall@5'] += recall_at_k(positive_probs, 5, ent_labels)
    
    
    def update_answer_metrics(self, metrics, prediction, gold):
        em = exact_match_score(prediction, gold)
        f1, prec, recall = f1_score(prediction, gold)
        metrics['ans_em'] += float(em)
        metrics['ans_f1'] += f1
        metrics['ans_prec'] += prec
        metrics['ans_recall'] += recall
        return em, prec, recall
    
    def update_joint_metrics(self, metrics, ans_em, ans_prec, ans_recall, sp_em, sp_prec, sp_recall):
        joint_prec = ans_prec * sp_prec
        joint_recall = ans_recall * sp_recall
        if joint_prec + joint_recall > 0:
            joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        else:
            joint_f1 = 0.
        joint_em =ans_em * sp_em
        metrics['joint_em'] += joint_em
        metrics['joint_f1'] += joint_f1
        metrics['joint_prec'] += joint_prec
        metrics['joint_recall'] += joint_recall


# %%
# model = HGNModel.from_pretrained('/workspace/ml-workspace/thesis_git/HSGN/models')
# model.cuda()

# %%
# validation = Validation(model, hotpot_dev, None, tokenizer,
#                         dev_tensor_input_ids, dev_tensor_attention_masks, 
#                         dev_tensor_token_type_ids,
#                         dev_list_span_idx)
# metrics = validation.do_validation()

# %%
# validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
#                         dev_tensor_input_ids, dev_tensor_attention_masks, 
#                         dev_tensor_token_type_ids,
#                         dev_list_span_idx)
# preds = validation.get_answer_predictions(None)
# with open('preds_no_wh_heuristics.json', 'w+') as f:
#     json.dump(preds, f)

# %%
# metrics

# %%
import os
import zipfile

def zipdir(path, name):
    zipf = zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()      


# %%
def record_eval_metric(neptune, metrics):
    for k, v in metrics.items():
        neptune.log_metric(k, v)


# %%
model_path = 'models'

best_eval_em = 0
# Measure the total training time for the whole run.
total_t0 = time.time()
with neptune.create_experiment(name="BertQA", params=PARAMS, upload_source_files=['src/models/GAT_Hierar_Tok_Node_Aggr.py']):
    neptune.set_property('server', 'IRGPU11')
    neptune.set_property('training_set_path', training_path)
    neptune.set_property('dev_set_path', dev_path)

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()
        # in the first epoch we use curriculum learning
        # in the second epoch we random the input to avoid biases (modifying the weights only for easy questions for a long time)
        if epoch_i > 0:
            random.shuffle(list_idx_curriculum_learning)
        # For each batch of training data...
        for step, idx in enumerate(tqdm(list_idx_curriculum_learning)):
            if list_span_idx[idx][0] == -1 or list_span_idx[idx][1] == -1:
                continue
            neptune.log_metric('step', step)
            # forward
            input_ids=tensor_input_ids[idx].unsqueeze(0).to(device)
            attention_mask=tensor_attention_masks[idx].unsqueeze(0).to(device)
            token_type_ids=tensor_token_type_ids[idx].unsqueeze(0).to(device) 
            start_positions=torch.tensor([list_span_idx[idx][0]], device='cuda')
            end_positions=torch.tensor([list_span_idx[idx][1]], device='cuda')
            output = model(None,
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids, 
                           start_positions=start_positions,
                           end_positions=end_positions)
            
            total_loss = output['span']['loss'] / dict_params['accumulation_steps']
            assert not torch.isnan(total_loss)
            
            # neptune
            neptune.log_metric("total_loss", total_loss.detach().item())

            # backpropagation
            total_loss.backward()
            if (step + 1) % dict_params['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                if (step +1) % 10000 == 0:
                    #############################
                    ######### Validation ########
                    #############################
                    validation = Validation(model, hotpot_dev, None, tokenizer,
                                            dev_tensor_input_ids, dev_tensor_attention_masks, 
                                            dev_tensor_token_type_ids,
                                            dev_list_span_idx)
                    metrics = validation.do_validation()
                    model.train()
                    record_eval_metric(neptune, metrics)

                    curr_em = metrics['ans_em']
                    if  curr_em > best_eval_em:
                        best_eval_em = curr_em
                        model.save_pretrained(model_path) 
            total_train_loss += total_loss.detach().item()
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(hotpot_train)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # #############################
        # ######### Validation ########
        # #############################
        validation = Validation(model, hotpot_dev, None, tokenizer,
                                dev_tensor_input_ids, dev_tensor_attention_masks, 
                                dev_tensor_token_type_ids,
                                dev_list_span_idx)
        metrics = validation.do_validation()
        model.train()
        record_eval_metric(neptune, metrics)

        curr_em = metrics['ans_em']
        if  curr_em > best_eval_em:
            best_eval_em = curr_em
            model.save_pretrained(model_path) 

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(hotpot_train)            

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

        
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    # create a zip file for the folder of the model
#     zipdir(model_path, os.path.join(model_path, 'checkpoint.zip'))
#     # upload the model to neptune
#     neptune.send_artifact(os.path.join(model_path, 'checkpoint.zip'))
