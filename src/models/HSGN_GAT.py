#!/usr/bin/env python
# coding: utf-8
# %%
import warnings
warnings.filterwarnings(action='once')


# %%
import json
import pickle
import os

data_path = "/workspace/ml-workspace/thesis_git/thesis/data/"
hotpot_qa_path = os.path.join(data_path, "external")
graphs_path = "processed/training/homog_20200804/"

with open(os.path.join(hotpot_qa_path, "hotpot_train_v1.1.json"), "r") as f:
    hotpot_train = json.load(f)
with open(os.path.join(hotpot_qa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
    hotpot_dev = json.load(f)


# %%
import os
os.environ['DGLBACKEND'] = 'pytorch'


# %%


from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

import dgl
from dgl.nn.pytorch.conv import GATConv


# %%


device = 'cuda'
pretrained_weights = 'bert-base-cased'


# %%


import random
random_seed = 2020
# Set the seed value all over the place to make this reproducible.
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


# ## HotpotQA Processing

# ## Processing

# # Process instance test

# %%


def graph_for_training(graph):
    tensor_node_type = graph.ndata['node_type']
    # shape [num_nodes, 1]
    sent_node_mask = tensor_node_type == 1
    # shape [num_nodes, 1]
    sent_nodes, _ = torch.nonzero(sent_node_mask, as_tuple=True)
    # contains the indexes of the sent nodes

    ent_node_mask = tensor_node_type == 3
    # shape [num_nodes, 1]
    ent_nodes, _ = torch.nonzero(ent_node_mask, as_tuple=True)
    # contains the indexes of the srl nodes

    sent_labels = graph.ndata['labels'][sent_nodes]
    # shape [num_sent_nodes, 1]
    ent_labels = graph.ndata['labels'][ent_nodes]

    return sum(sent_labels).item() > 0 and sum(ent_labels).item() > 0


# %%


def graph_for_training(graph):
    tensor_node_type = graph.ndata['node_type']
    # shape [num_nodes, 1]
    sent_node_mask = tensor_node_type == 1
    # shape [num_nodes, 1]
    sent_nodes, _ = torch.nonzero(sent_node_mask, as_tuple=True)
    # contains the indexes of the sent nodes
    sent_labels = graph.ndata['labels'][sent_nodes]
    # shape [num_sent_nodes, 1]
    
    ent_node_mask = tensor_node_type == 3
    # shape [num_nodes, 1]
    ent_nodes, _ = torch.nonzero(ent_node_mask, as_tuple=True)
    # contains the indexes of the ent nodes
    if len(ent_nodes) > 0:
        ent_labels = graph.ndata['labels'][ent_nodes]
        return sum(sent_labels).item() > 0 and sum(ent_labels).item() > 0
    return False #skip for now graphs without entities


# %%

data_path = "/workspace/ml-workspace/thesis_git/thesis/data/"
training_path = os.path.join(data_path, "processed/training/homog_20200804/")
dev_path = os.path.join(data_path, "processed/dev/homog_20200804/")

with open(os.path.join(training_path, 'list_span_idx.p'), 'rb') as f:
    list_span_idx = pickle.load(f)


# %%


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


from os import listdir
from os.path import isfile, join
from dgl.data.utils import load_graphs


# %%


import re
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# %%
training_graphs_path = os.path.join(training_path, 'graphs')
list_graph_files = natural_sort([f for f in listdir(training_graphs_path) if isfile(join(training_graphs_path, f))])
list_graphs = []
for g_file in tqdm(list_graph_files):
    if ".bin" in g_file:
        with open(os.path.join(training_graphs_path, g_file), 'rb') as f:
            graph = pickle.load(f)
        list_graphs.append(graph)


# %%

dev_graphs_path = os.path.join(dev_path, 'graphs')
list_graph_files = natural_sort([f for f in listdir(dev_graphs_path) if isfile(join(dev_graphs_path, f))])
dev_list_graphs = []
for g_file in tqdm(list_graph_files):
    if ".bin" in g_file:
        with open(os.path.join(dev_graphs_path, g_file), 'rb') as f:
            graph = pickle.load(f)
        dev_list_graphs.append(graph)


# # BERT GAT

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


# %%


import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads = 1,
                 feat_drop = 0.1,
                 attn_drop = 0.1,
                 negative_slope = None,
                 residual = True,
                 activation = None):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(in_dim, num_hidden, num_heads,
                                             feat_drop = feat_drop,
                                             attn_drop =attn_drop, 
                                             residual= residual,
                                             activation=activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv( num_hidden * num_heads, num_hidden, num_heads,
                                             feat_drop = feat_drop,
                                             attn_drop =attn_drop, 
                                             residual= residual,
                                             activation=activation))
        # output projection
        self.gat_layers.append(GATConv( num_hidden * num_heads, num_hidden, num_heads,
                                             feat_drop = feat_drop,
                                             attn_drop =attn_drop, 
                                             residual= residual,
                                             activation=activation))

    def forward(self, g, h):
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


# %%


dict_params = {'in_feats': 768, 'out_feats': 768, 'feat_drop': 0.1, 'attn_drop': 0.1, 'hidden_size_classifier': 768,
               'weight_sent_loss': 1, 'weight_srl_loss': 1, 'weight_ent_loss': 1,
               'weight_span_loss': 1, 'weight_ans_type_loss': 1, 
               'gat_layers': 2}
class HGNModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # graph
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.srl_emb_gat = GATConv(dict_params['in_feats'], dict_params['out_feats'],
                           2, feat_drop = dict_params['feat_drop'],
                           attn_drop = dict_params['attn_drop'], 
                            residual=True)
        self.gat = GAT(dict_params['gat_layers'], dict_params['in_feats'], dict_params['in_feats'], dict_params['out_feats'],
                           2, feat_drop = dict_params['feat_drop'],
                           attn_drop = dict_params['attn_drop'])
        ## node classification
        ### ent node
        self.dropout_ent = nn.Dropout(config.hidden_dropout_prob)
        self.ent_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                      dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        ### srl node
        self.dropout_srl = nn.Dropout(config.hidden_dropout_prob)
        self.srl_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                      dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        ### sent node
        self.dropout_sent = nn.Dropout(config.hidden_dropout_prob)
        self.sent_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                       dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        # graph 2 token attention
        self.graph2token_attention = GAT(1, dict_params['in_feats'], 
                                               dict_params['in_feats'], 
                                               dict_params['out_feats'])
        # span prediction
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # init weights
        self.init_weights()
        # params
        self.weight_sent_loss = dict_params['weight_sent_loss']
        self.weight_srl_loss = dict_params['weight_srl_loss']
        self.weight_ent_loss = dict_params['weight_ent_loss']
        self.weight_span_loss = dict_params['weight_span_loss']
    
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
        assert not torch.isnan(sequence_output).any()
        # Graph forward & node classification
        graph_out, graph_emb = self.graph_forward(graph, sequence_output, train)
        sequence_output = graph_emb[0:512].unsqueeze(0)
        # add graph info to the bert token embeddings
        #sequence_output = self.update_sequence_outputs(sequence_output, graph, graph_emb)

        # span prediction
        span_loss, start_logits, end_logits = self.span_prediction(sequence_output, start_positions, end_positions)
        assert not torch.isnan(start_logits).any()
        assert not torch.isnan(end_logits).any()
        # loss
        final_loss = 0.0
        if span_loss != 0:
            final_loss += self.weight_span_loss*span_loss
        if graph_out['sent']['loss'] != 0 :
            final_loss += self.weight_sent_loss*graph_out['sent']['loss']
        if graph_out['srl']['loss'] != 0:
            final_loss += self.weight_srl_loss*graph_out['srl']['loss']
        if graph_out['ent']['loss'] != 0:
            final_loss += self.weight_ent_loss*graph_out['ent']['loss']
        
        return {'loss': final_loss, 
                'sent': graph_out['sent'], 
                'ent': graph_out['ent'],
                'srl': graph_out['srl'],
                'span': {'loss': span_loss, 'start_logits': start_logits, 'end_logits': end_logits}}  
    
    def graph_forward(self, graph, bert_context_emb, train):
        # create graph initial embedding #
        graph_emb = self.graph_initial_embedding(graph, bert_context_emb)
        assert not torch.isnan(graph_emb).any()
        # graph_emb shape [num_nodes, in_feats]    
        sample_sent_nodes = self.sample_sent_nodes(graph)
        sample_srl_nodes = self.sample_srl_nodes(graph)
        sample_ent_nodes = self.sample_ent_nodes(graph)
        initial_graph_emb = graph_emb # for skip-connection
        
        # update graph embedding #
        graph_emb = self.gat(graph, graph_emb)
        # graph_emb shape [num_nodes, num_heads, in_feats] num_heads = 1
        graph_emb = graph_emb.view(-1, dict_params['out_feats'])
        # graph_emb shape [num_nodes, in_feats]

        # classify nodes #
        tensor_node_type = graph.ndata['node_type']
        # sent node idx
        sent_node_mask = tensor_node_type == 1
        # shape [num_nodes, 1] # contains the indexes of the sent nodes
        sent_nodes, _ = torch.nonzero(sent_node_mask, as_tuple=True)
        
        # srl node idx
        srl_node_mask = tensor_node_type == 2
        srl_nodes, _ = torch.nonzero(srl_node_mask, as_tuple=True)
        # shape [num_nodes, 1] # contains the indexes of the srl nodes
        
        # ent node idx
        ent_node_mask = tensor_node_type == 3
        ent_nodes, _ = torch.nonzero(ent_node_mask, as_tuple=True)
        
        # shape [num_nodes, 1]
        # loss #
        #loss_fn = nn.CrossEntropyLoss()
        
        sent_labels = None
        ent_labels = None
        if train:
            # add skip-connection
            logits_sent = self.sent_classifier(torch.cat((graph_emb[sent_nodes][sample_sent_nodes],
                                                          initial_graph_emb[sent_nodes][sample_sent_nodes]), dim=1))
            assert not torch.isnan(logits_sent).any() 
            
            # contains the indexes of the srl nodes
            if len(sample_srl_nodes) == 0:
                logits_srl = torch.tensor([[]], device=device)
            else:
                logits_srl = self.srl_classifier(torch.cat((graph_emb[srl_nodes][sample_srl_nodes],
                                                            initial_graph_emb[srl_nodes][sample_srl_nodes]), dim=1))
                # shape [num_ent_nodes, 2] 
                assert not torch.isnan(logits_srl).any()
            
            # contains the indexes of the ent nodes
            if len(sample_ent_nodes) == 0:
                logits_ent = torch.tensor([[]], device=device)
            else:
                logits_ent = self.ent_classifier(torch.cat((graph_emb[ent_nodes][sample_ent_nodes],
                                                            initial_graph_emb[ent_nodes][sample_ent_nodes]), dim=1))
                # shape [num_ent_nodes, 2] 
                assert not torch.isnan(logits_ent).any()
                       
            sent_labels = graph.ndata['labels'][sent_nodes][sample_sent_nodes].to(device)
            # shape [num_sampled_sent_nodes, 1]
            srl_labels = graph.ndata['labels'][srl_nodes][sample_srl_nodes].to(device)
            # shape [num_sampled_srl_nodes, 1]
            ent_labels = graph.ndata['labels'][ent_nodes][sample_ent_nodes].to(device)
            # shape [num_sampled_ent_nodes, 1]
        else:
            # add skip-connection
            logits_sent = self.sent_classifier(torch.cat((graph_emb[sent_nodes],
                                                          initial_graph_emb[sent_nodes]), dim=1))
            assert not torch.isnan(logits_sent).any()
            logits_srl = self.srl_classifier(torch.cat((graph_emb[srl_nodes],
                                                        initial_graph_emb[srl_nodes]), dim=1))
            # shape [num_ent_nodes, 2] 
            assert not torch.isnan(logits_srl).any()
            
            logits_ent = self.ent_classifier(torch.cat((graph_emb[ent_nodes],
                                                        initial_graph_emb[ent_nodes]), dim=1))
            # shape [num_ent_nodes, 2]
            assert not torch.isnan(logits_ent).any()
            
            # labels
            sent_labels = graph.ndata['labels'][sent_nodes].to(device)
            # shape [num_sent_nodes, 1]
            srl_labels = graph.ndata['labels'][srl_nodes].to(device)
            # shape [num_sampled_srl_nodes, 1]
            ent_labels = graph.ndata['labels'][ent_nodes].to(device)
            # shape [num_srl_nodes, 1]
           
        # sent loss
        loss_sent = loss_fn(logits_sent, sent_labels.view(-1))
        probs_sent = F.softmax(logits_sent, dim=1).cpu()
        # shape [num_sent_nodes, 2]
        
        # srl loss
        loss_srl = 0.0 # not all ans are inside an srl arg
        probs_ent = torch.tensor([], device=device)
        if len(logits_srl) != 0:
            loss_srl = loss_fn(logits_srl, srl_labels.view(-1))
            probs_srl = F.softmax(logits_srl, dim=1).cpu()
            # shape [num_srl_nodes, 2]
        # ent loss
        loss_ent = 0.0 # not all ans are an entity
        probs_ent = torch.tensor([], device=device)
        if len(logits_ent) != 0:
            loss_ent = loss_fn(logits_ent, ent_labels.view(-1))
            probs_ent = F.softmax(logits_ent, dim=1).cpu()
            # shape [num_ent_nodes, 2]

        sent_labels = sent_labels.cpu()
        ent_labels = ent_labels.cpu()

        return ({'sent': {'loss': loss_sent, 'probs': probs_sent, 'lbl': sent_labels.view(-1)},
                'srl': {'loss': loss_srl, 'probs': probs_srl, 'lbl': srl_labels.view(-1)},
                'ent': {'loss': loss_ent, 'probs': probs_ent, 'lbl': ent_labels.view(-1)}},
                graph_emb)
    
    def graph_initial_embedding(self, graph, bert_context_emb):
        '''
        Inputs:
            - graph
            - bert_context_emb shape [1, #max len, 768]
        '''
        graph_emb = torch.zeros((graph.number_of_nodes(), 768), device=device)
        for node, (st, end) in enumerate(graph.ndata['st_end_idx']):
            graph_emb[node] = self.aggregate_emb(bert_context_emb[0][st:end])
        return graph_emb
    
    def aggregate_emb(self, token_emb):
        # average for now
        return torch.mean(token_emb, dim = 0)
    
    def sample_sent_nodes(self, graph):
        list_sent_nodes = [node for node, ntype in enumerate(graph.ndata['node_type']) if ntype == 1]
        sent_nodes_labels = graph.ndata['labels'][list_sent_nodes]
        # shape [num sent nodes x 1]
        supporting_sent_mask = sent_nodes_labels == torch.ones((sent_nodes_labels.shape))
        # shape [num sent nodes x 1] with values True or False
        supporting_sent_idx = [idx for idx, class_ in enumerate(supporting_sent_mask) if class_]
        # list with the idx of supporting sent
        non_supp_sent_idx = [i for i, non_supp_sent_node in enumerate(~supporting_sent_mask) if non_supp_sent_node]
        # list with the idx of non supporting sent
        num_supp_sent = len(supporting_sent_idx)
        non_supp_sent_idx_sample = random.sample(non_supp_sent_idx, min(num_supp_sent, 
                                                                        len(non_supp_sent_idx)))
        sent_sample_idx = supporting_sent_idx + non_supp_sent_idx_sample
        sent_sample_idx.sort()

        return sent_sample_idx

    def sample_srl_nodes(self, graph):
        list_srl_nodes = [node for node, ntype in enumerate(graph.ndata['node_type']) if ntype == 2]
        srl_nodes_labels = graph.ndata['labels'][list_srl_nodes]
        # shape [num labels x 1]
        supporting_srl_mask = srl_nodes_labels == torch.ones((srl_nodes_labels.shape))
        # shape [num labels x 1] with values True or False
        supporting_srl_idx = [idx for idx, class_ in enumerate(supporting_srl_mask) if class_]
        # list with the idx of supporting srl
        non_supp_srl_idx = [i for i, non_supp_srl_node in enumerate(~supporting_srl_mask) if non_supp_srl_node]
        # list with the idx of non supporting srl
        num_supp_srl = len(supporting_srl_idx)
        non_supp_srl_idx_sample = random.sample(non_supp_srl_idx, min(num_supp_srl, 
                                                                        len(non_supp_srl_idx)))
        srl_sample_idx = supporting_srl_idx + non_supp_srl_idx_sample
        srl_sample_idx.sort()

        return srl_sample_idx
    
    def sample_ent_nodes(self, graph):
        list_ent_nodes = [node for node, ntype in enumerate(graph.ndata['node_type']) if ntype == 3]
        ent_nodes_labels = graph.ndata['labels'][list_ent_nodes]
        # shape [num labels x 1]
        supporting_ent_mask = ent_nodes_labels == torch.ones((ent_nodes_labels.shape))
        # shape [num labels x 1] with values True or False
        supporting_ent_idx = [idx for idx, class_ in enumerate(supporting_ent_mask) if class_]
        # list with the idx of supporting srl
        non_supp_ent_idx = [i for i, non_supp_ent_node in enumerate(~supporting_ent_mask) if non_supp_ent_node]
        # list with the idx of non supporting srl
        num_supp_ent = len(supporting_ent_idx)
        non_supp_ent_idx_sample = random.sample(non_supp_ent_idx, min(num_supp_ent, 
                                                                      len(non_supp_ent_idx)))
        ent_sample_idx = supporting_ent_idx + non_supp_ent_idx_sample
        ent_sample_idx.sort()

        return ent_sample_idx
    
    
    def update_sequence_outputs(self, sequence_output, graph, graph_emb):
        list_edges = []
        offset_node = sequence_output.shape[1]
        for node_idx, (st, end) in enumerate(graph.ndata['st_end_idx']):
            u = node_idx + offset_node
            list_edges.extend([(u, t) for t in range(st, end)])
        # embeddign of the new graph: bert token embs + HS graph
        g2d_graph_emb = torch.cat((sequence_output.squeeze(0), graph_emb))
        # graph creation for the self attention
        g2d_graph = dgl.DGLGraph()
        g2d_graph.add_nodes(g2d_graph_emb.shape[0])
        src, dst = tuple(zip(*list_edges))
        # dgl is directional
        g2d_graph.add_edges(src, dst)
        g2d_graph_emb = self.graph2token_attention(g2d_graph, g2d_graph_emb)
        return g2d_graph_emb[0:offset_node].unsqueeze(0)
    
    def span_prediction(self, sequence_output, start_positions, end_positions):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = 0.0
        if start_positions is not None and end_positions is not None:
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


train_dataloader = list_graphs


# %%


lr = 1e-5
optimizer = AdamW(model.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5, 
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


# %%


from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
epochs = 1

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

#Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                             num_warmup_steps = 0, # Default value in run_glue.py
                                             num_training_steps = total_steps)

# scheduler_medium = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = len(train_dataloader_medium) * epochs)
# scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
#                                                                num_warmup_steps = 0, # Default value in run_glue.py
#                                                                num_training_steps = total_steps)


# # Neptune Config

# %%


neptune_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYTA2MTgwYjQtMGJkMS00MTcxLTk0MWEtZjIxZThmYjlhYTA5In0="


# %%


train_batch_size = 1


# %%


import neptune
neptune.init(
    "haritz/srl-pred",
    api_token=neptune_token,
)
neptune.set_project('haritz/srl-pred')
PARAMS = {"num_epoch": epochs, 
          'lr': lr, 
          'pretrained_weights': pretrained_weights,
          'loss_fn': 'crossentropy_label_smoothing', 
          #'validation_size': len(validation_dataloader)*val_batch_size , 
          'random_seed': random_seed,
          'total_steps': total_steps, 
          'training_size': len(train_dataloader)*train_batch_size, 
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


def get_pred_ans_str(input_ids, output, tokenizer):
    st = torch.argmax(output['span']['start_logits'], dim=1).item()
    end = torch.argmax(output['span']['end_logits'], dim=1).item()
    return tokenizer.decode(input_ids[st:end])


# %%


tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


# %%


class Validation():

    def __init__(self, model, dataset, validation_dataloader, tokenizer,
                 tensor_input_ids, tensor_attention_masks, tensor_token_type_ids, list_span_idx):
        self.model = model
        self.model.eval()
        self.dataset = dataset
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
                   'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0, 
                   'srl_em': 0, 'srl_f1': 0, 'srl_prec': 0, 'srl_recall': 0,
                   'srl_recall@1': 0, 'srl_recall@3': 0, 'srl_recall@5': 0,
                   'ent_em': 0, 'ent_f1': 0, 'ent_prec': 0, 'ent_recall': 0,
                   'ent_recall@1': 0, 'ent_recall@3': 0, 'ent_recall@5': 0,
                   }
        # Evaluate data for one epoch       
        num_valid_examples = 0
        for step, b_graph in enumerate(tqdm(self.validation_dataloader)):
            if not graph_for_training(b_graph) or self.list_span_idx[step] == (-1, -1):
                continue
            num_valid_examples += 1
            
            with torch.no_grad():
                output = self.model(b_graph,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0), 
                               start_positions=torch.tensor([self.list_span_idx[step][0]], device='cuda'),
                               end_positions=torch.tensor([self.list_span_idx[step][1]], device='cuda'), train=False)
                
            # Accumulate the validation loss.
            metrics['validation_loss'] += output['loss'].item()
            # Sentence evaluation
            sent_labels = output['sent']['lbl']
            prediction_sent = torch.argmax(output['sent']['probs'], dim=1)
            sp_em, sp_prec, sp_recall = self.update_sp_metrics(metrics, prediction_sent, sent_labels)
            # srl
            prediction_srl = torch.argmax(output['srl']['probs'], dim=1)
            srl_labels = output['srl']['lbl']
            self.update_srl_metrics(metrics, prediction_srl, srl_labels, output['srl']['probs'][:,1])
            # ent
            no_ent = False
            if len(output['ent']['probs']) != 0:
                prediction_ent = torch.argmax(output['ent']['probs'], dim=1)
            else:
                no_ent = True
            if not no_ent:
                ent_labels = output['ent']['lbl']
                self.update_ent_metrics(metrics, prediction_ent, ent_labels, output['ent']['probs'][:,1])
            #span prediction
            golden_ans = self.dataset[step]['answer']
            predicted_ans = get_pred_ans_str(self.tensor_input_ids[step], output, tokenizer)
            ans_em, ans_prec, ans_recall = self.update_answer_metrics(metrics, predicted_ans, golden_ans)
            # joint
            self.update_joint_metrics(metrics, ans_em, ans_prec, ans_recall, sp_em, sp_prec, sp_recall)                

        #N = len(self.validation_dataloader)
        N = num_valid_examples
        for k in metrics.keys():
            metrics[k] /= N
        return metrics
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
model_path = '/workspace/ml-workspace/thesis_git/thesis/models/'
best_eval_f1 = 0
# Measure the total training time for the whole run.
total_t0 = time.time()
with neptune.create_experiment(name="HierarchicalSemanticGraphNetwork", params=PARAMS, upload_source_files=['HSGN_GAT.py']):
    neptune.append_tag(["homogeneous_graph", "GATConv", "bidirectional_token_node_edge"])
    neptune.set_property('server', 'IRGPU2')
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

        # For each batch of training data...
        for step, b_graph in enumerate(tqdm(list_graphs)):
            if not graph_for_training(list_graphs[step]) or list_span_idx[step] == (-1, -1):
                continue
            model.zero_grad()  
            # forward
            output = model(list_graphs[step],
                           input_ids=tensor_input_ids[step].unsqueeze(0),
                           attention_mask=tensor_attention_masks[step].unsqueeze(0),
                           token_type_ids=tensor_token_type_ids[step].unsqueeze(0), 
                           start_positions=torch.tensor([list_span_idx[step][0]], device='cuda'),
                           end_positions=torch.tensor([list_span_idx[step][1]], device='cuda'))
            
            total_loss = output['loss']
            assert not torch.isnan(total_loss)
            sent_loss = output['sent']['loss']
            ent_loss = output['ent']['loss']
            srl_loss = output['srl']['loss']
            span_loss = output['span']['loss']
            neptune.log_metric("total_loss", total_loss.detach().item())
            neptune.log_metric("sent_loss", sent_loss.detach().item())
            neptune.log_metric("srl_loss", srl_loss.detach().item())
            neptune.log_metric("ent_loss", ent_loss.detach().item())
            neptune.log_metric("span_loss", span_loss.detach().item())

            total_train_loss += total_loss.item()
            # backpropagation
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 4000 == 0 and step != 0:
                #############################
                ######### Validation ########
                #############################
                validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
                                        dev_tensor_input_ids, dev_tensor_attention_masks, 
                                        dev_tensor_token_type_ids,
                                        dev_list_span_idx)
                metrics = validation.do_validation()
                model.train()
                record_eval_metric(neptune, metrics)

                curr_f1 = metrics['joint_f1']
                if  curr_f1 > best_eval_f1:
                    best_eval_f1 = curr_f1
                    model.save_pretrained(model_path) 

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # #############################
        # ######### Validation ########
        # #############################
        validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
                                dev_tensor_input_ids, dev_tensor_attention_masks, 
                                dev_tensor_token_type_ids,
                                dev_list_span_idx)
        metrics = validation.do_validation()
        model.train()
        record_eval_metric(neptune, metrics)

        curr_f1 = metrics['joint_f1']
        if  curr_f1 > best_eval_f1:
            best_eval_f1 = curr_f1
            model.save_pretrained(model_path) 

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

        
    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    # create a zip file for the folder of the model
    zipdir(model_path, os.path.join(model_path, 'checkpoint.zip'))
    # upload the model to neptune
    neptune.send_artifact(os.path.join(model_path, 'checkpoint.zip'))
