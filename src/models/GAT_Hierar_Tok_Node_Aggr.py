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
pretrained_weights = 'bert-base-uncased'
#pretrained_weights = 'bert-large-cased-whole-word-masking'
#pretrained_weights = 'albert-xxlarge-v2'
# ## HotpotQA Processing

# ## Processing

# %%
training_path = os.path.join(data_path, "processed/training/hsgn_2021_fix/")
dev_path = os.path.join(data_path, "processed/dev/hsgn_2021_fix/")

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
def add_metadata2graph(graph, metadata):
    for (node, dict_node) in metadata.items():
        for (k, v) in dict_node.items():
            if node in graph.ntypes:
                graph.nodes[node].data[k] = torch.tensor(v)
    return graph


# %%
training_graphs_path = os.path.join(training_path, 'graphs')
training_metadata_path = os.path.join(training_path, 'metadata')

list_graph_files = natural_sort([f for f in listdir(training_graphs_path) if isfile(join(training_graphs_path, f))])
list_metadata_files = natural_sort([f for f in listdir(training_metadata_path) if isfile(join(training_metadata_path, f))])
list_graph_metadata_files = list(zip(list_graph_files, list_metadata_files))

list_graphs = []
for (g_file, metadata_file) in tqdm(list_graph_metadata_files):
    if ".bin" in g_file:
        with open(os.path.join(training_graphs_path, g_file), "rb") as f:
            graph = pickle.load(f)
        with open(os.path.join(training_metadata_path, metadata_file), "rb") as f:
            metadata = pickle.load(f)
        # add metadata to the graph
        graph = add_metadata2graph(graph, metadata)
        list_graphs.append(graph)

# %%
dev_graphs_path = os.path.join(dev_path, 'graphs/')
dev_metadata_path = os.path.join(dev_path, 'metadata/')

list_graph_files = natural_sort([f for f in listdir(dev_graphs_path) if isfile(join(dev_graphs_path, f))])
list_metadata_files = natural_sort([f for f in listdir(dev_metadata_path) if isfile(join(dev_metadata_path, f))])
list_graph_metadata_files = list(zip(list_graph_files, list_metadata_files))

dev_list_graphs = []
for (g_file, metadata_file) in tqdm(list_graph_metadata_files):
    if ".bin" in g_file:
        with open(os.path.join(dev_graphs_path, g_file), "rb") as f:
            graph = pickle.load(f)
        with open(os.path.join(dev_metadata_path, metadata_file), "rb") as f:
            metadata = pickle.load(f)
        # add metadata to the graph
        graph = add_metadata2graph(graph, metadata)
        dev_list_graphs.append(graph)

# %%
# def graph_for_eval(graph):
#     return ((sum(graph.nodes['sent'].data['labels']) > 0) and
#             (sum(graph.nodes['srl'].data['labels']) > 0) and
#             (sum(graph.nodes['ent'].data['labels']) > 0 )
#            )

# %%
weights = [0., 0., 0.]
for ins in hotpot_train:
    if ins['answer'] == 'yes':
        weights[1] += 1
    elif ins['answer'] == 'no':
        weights[2] += 1
    else:
        weights[0] += 1
weights = max(weights) / torch.tensor(weights, device=device)

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
#loss_fn = LabelSmoothingLoss()
loss_fn = nn.CrossEntropyLoss()
# %%
# https://github.com/miafei/NodeNorm
class NodeNorm(nn.Module):
    def __init__(self, unbiased=False, eps=1e-5):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x

# %%
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
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, feat_drop=0., attn_drop=0.):
        super(HeteroRGCNLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.rel_trans = nn.Linear(in_size + bert_dim, in_size)
        self.node_trans = nn.Sequential(nn.Linear(in_size, in_size),
                                    nn.LeakyReLU(),
                                        nn.Dropout(feat_drop),
                                        nn.Linear(in_size, out_size))
        self.node_att = nn.Linear(2 * out_size, out_size)       
        self.gru_node2tok = nn.GRU(out_size, out_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.reset_parameters()

    def message_func_rel(self, bert_token_emb, edges):
        '''
        m_ij = R_ji * W * h_j
        '''
        # relation emb
        rel_span_idx = edges.data['span_idx']  # idx at context level
        rel_emb = []
        for i, (x, y) in enumerate(rel_span_idx):
            # rel type = {1, -1}
            rel_emb.append(torch.mean(bert_token_emb[x:y], dim=0))
        rel_emb = torch.stack(rel_emb, dim=0)
        assert not torch.isnan(rel_emb).any()
#         return {'rel': rel_emb, 'srl': edges.src['h']}
        src = edges.src['h']
        m = self.node_trans(self.rel_trans(torch.cat((src, rel_emb), dim=1)))
        # m: [num srl x 768]
        dst = self.node_trans(edges.dst['h'])
        cat_uv = torch.cat([m,
                            dst],
                           dim=1)
        e = F.leaky_relu(self.node_att(cat_uv))
        return {'m': m, 'e': e}

    def reduce_func(self, nodes):
        '''
        h_srl = sum_j(h_j) + h_srl # w/o transformation for h_srl for now
        
        '''       
        alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h': h}
    
    def message_func_2tok(self, edges):
#         src = edges.src['h'].view(1,-1,self.in_size)
#         tok = edges.dst['h'].view(1,-1,self.in_size) # the token node
#         m = self.gru_node2tok(src, tok)[0].squeeze(0)
        src = edges.src['h'] # already updated 
        return {'m': src}
    
    def reduce_func_srl2tok(self, nodes):
        h = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h_srl': h}
    
    def reduce_func_ent2tok(self, nodes):
        h = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h_ent': h}
    
    def message_func_regular_node(self, edges):
        '''
        e_ij = alpha_ij * W * h_j
        alpha_ij = LeakyReLU(W * (Wh_j || Wh_i))
        '''
        src = edges.src['h']
        updt_dst = self.node_trans(edges.dst['h'])
        updt_src = self.node_trans(src)
        cat_uv = torch.cat([updt_src,
                            updt_dst],
                           dim=1)
        e = F.leaky_relu(self.node_att(cat_uv))
        return {'m': updt_src, 'e': e}

    def forward(self, G, feat_dict, bert_token_emb):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if 'h' not in G.nodes[srctype].data:
                G.nodes[srctype].data['h'] = self.feat_drop(feat_dict[srctype])
            if 'h' not in G.nodes[dsttype].data:
                G.nodes[dsttype].data['h'] = self.feat_drop(feat_dict[dsttype])

            if "2tok" in etype:
                pass
            elif "srl2srl" == etype:
                funcs[etype] = ((lambda e: self.message_func_rel(bert_token_emb, e)) , self.reduce_func)
            elif "ent2ent_rel" == etype:
                funcs[etype] = ((lambda e: self.message_func_rel(bert_token_emb, e)) , self.reduce_func)
            else:
                funcs[etype] = (self.message_func_regular_node, self.reduce_func)
        G.multi_update_all(funcs, 'sum')
        ## update tokens
        G['srl2tok'].update_all(self.message_func_2tok, fn.sum('m', 'h_srl'))
        if 'ent' in G.ntypes:
            G['ent2tok'].update_all(self.message_func_2tok, fn.sum('m', 'h_ent'))
        #batched all tokens since we want to put into the GRU (srl, ent, hidden=tok) so batch size = 512
        h_tok = self.node_trans(G.nodes['tok'].data['h'])
        h_tok = h_tok.view(1,-1,self.out_size)
        h_srl = G.nodes['tok'].data.pop('h_srl').view(1,-1,self.out_size)
        gru_input = torch.cat((h_srl, h_tok), dim=0)  
        if 'h_ent' in G.nodes['tok'].data:
            # there can be an instance without entities (not common anyway)
            h_ent = G.nodes['tok'].data.pop('h_ent').view(1,-1,self.out_size)
            gru_input = torch.cat((h_ent, h_tok), dim=0)
            gru_input = torch.cat((h_srl, gru_input), dim=0)  
        G.nodes['tok'].data['h'] = self.gru_node2tok(gru_input)[0][-1]

        out = {ntype : G.nodes[ntype].data.pop('h') for ntype in G.ntypes}
        # return the updated node feature dictionary
        self.clean_memory(G)
        return out

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.node_trans[0].weight, gain=gain)
        nn.init.xavier_normal_(self.node_trans[3].weight, gain=gain)
        nn.init.xavier_normal_(self.node_att.weight, gain=gain)
        for param in self.gru_node2tok.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def clean_memory(self, graph):
        # remove garbage from the graph computation
        node_tensors = ['h']
        for ntype in graph.ntypes:
            for key in node_tensors:
                if key in graph.nodes[ntype].data.keys():
                    del graph.nodes[ntype].data[key]

        edges_tensors = ['e', 'm']
        for (_, etype, _) in graph.canonical_etypes:
            for key in edges_tensors:
                if key in graph.edges[etype].data.keys():
                    del graph.edges[etype].data[key]


# %%
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_size, out_size, feat_drop, attn_drop):
        super(MultiHeadGATLayer, self).__init__()
        self.in_size = in_size
        self.node_norm = NodeNorm()
        self.head1 = HeteroRGCNLayer(in_size, out_size, feat_drop, attn_drop)
        self.head2 = HeteroRGCNLayer(in_size, out_size, feat_drop, attn_drop)

    def forward(self, G, emb, bert_token_emb):
        h_dict1 = self.head1(G, emb, bert_token_emb)
        h_dict2 = self.head2(G, emb, bert_token_emb)
        cat_h_dict = dict()
        for k in h_dict1.keys():
            cat_h_dict[k] = torch.cat((h_dict1[k], h_dict2[k]), dim=1)
        # concat on the output feature dimension (dim=1)
        return cat_h_dict
# %%
class HeteroRGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, feat_drop, attn_drop, residual):
        super(HeteroRGCN, self).__init__()
        self.in_size = in_size
        self.residual = residual
        self.node_norm = NodeNorm()
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, feat_drop, attn_drop)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, feat_drop, attn_drop)
        self.gru_layer_lvl = nn.GRU(in_size, out_size)
        
        self.init_params()

    def forward(self, G, emb, bert_token_emb):
        #h_tok0 = emb['tok'].view(1,-1,self.in_size) # it's already normalized
        h_dict0 = {k: self.node_norm(h) for k, h in emb.items()}
        
        h_dict1 = self.layer1(G, h_dict0, bert_token_emb)
        #h_tok1 = self.node_norm(h_dict['tok'].view(1,-1,self.in_size))
        h_dict1 = {k: F.leaky_relu(self.node_norm(h)) for k, h in h_dict1.items()}
        
        h_dict2 = self.layer2(G, h_dict1, bert_token_emb)
        h_dict2 = {k: F.leaky_relu(self.node_norm(h)) for k, h in h_dict2.items()}
        #h_tok2 = self.node_norm(h_dict['tok'].view(1,-1,self.in_size))
        h_final = h_dict2
        if self.residual:
            # h_dict = {k : F.leaky_relu(self.node_norm(h_dict[k]) + h_dict0[k]) for k in h_dict.keys()}
            h_final = dict()
            for k in h_dict0.keys():
                gru_input = torch.cat((h_dict1[k].view(1,-1,self.in_size), h_dict2[k].view(1,-1,self.in_size)), dim=0)
                h_final[k] = self.gru_layer_lvl(gru_input, h_dict0[k].view(1,-1,self.in_size))[0][-1].view(-1, self.in_size)
        return h_final
    
    def init_params(self):
        for param in self.gru_layer_lvl.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

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
               'weight_sent_loss': 2, 'weight_srl_loss': 1, 'weight_ent_loss': 1, 'bi_gru_layers': 1,
               'weight_span_loss': 5, 'weight_ans_type_loss': 1, 'span_drop': 0.2,
               'gat_layers': 4, 'accumulation_steps': 1, 'residual': True,}
class HGNModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if pretrained_weights == 'albert-xxlarge-v2':
            self.bert = AlbertModel(config)
        else:
            self.bert = BertModel(config)
        # Initial Node Embedding
        self.bigru = nn.GRU(input_size=dict_params['in_feats'], hidden_size=dict_params['in_feats'], 
                            num_layers=dict_params['bi_gru_layers'], dropout = dict_params['feat_drop'], bidirectional=True)
        self.gru_aggregation = nn.Linear(2*dict_params['in_feats'], dict_params['in_feats'])
        self.node_norm = NodeNorm()
        # Graph Neural Network
        self.rgcn = HeteroRGCN(dict_params['in_feats'], dict_params['in_feats'],
                               dict_params['out_feats'], dict_params['feat_drop'], dict_params['attn_drop'], 
                               dict_params['residual'])
        ## node classification
        ### ent node
        self.ent_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                      dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        ### srl node
        self.srl_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                      dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        ### sent node
        self.sent_classifier = nn.Sequential(nn.Linear(2*dict_params['out_feats'],
                                                       dict_params['hidden_size_classifier']),
                                            nn.ReLU(),
                                            nn.Linear(dict_params['hidden_size_classifier'],
                                                      2))
        
        # span prediction
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        GeLU(),
                                        nn.Dropout(dict_params['span_drop']),
                                        nn.Linear(config.hidden_size, 2))
        
        # Answer Type
        self.sfm = nn.Softmax(-1)
        self.answer_type_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), 
                                                    GeLU(),
                                                    nn.Dropout(dict_params['span_drop']),
                                                    nn.Linear(config.hidden_size, 3)) 
        # init weights
        
        # params
        self.weight_sent_loss = dict_params['weight_sent_loss']
        self.weight_srl_loss = dict_params['weight_srl_loss']
        self.weight_ent_loss = dict_params['weight_ent_loss']
        self.weight_span_loss = dict_params['weight_span_loss']
        self.weight_ans_type_loss = dict_params['weight_ans_type_loss']
    
    def init_params(self):
        self.init_weights()
        nn.init.xavier_normal_(self.sent_classifier[0].weight)
        nn.init.xavier_normal_(self.sent_classifier[2].weight)

        nn.init.xavier_normal_(self.ent_classifier[0].weight)
        nn.init.xavier_normal_(self.ent_classifier[2].weight)

        nn.init.xavier_normal_(self.srl_classifier[0].weight)
        nn.init.xavier_normal_(self.srl_classifier[2].weight)

        nn.init.xavier_normal_(self.answer_type_classifier[0].weight)
        nn.init.xavier_normal_(self.answer_type_classifier[3].weight)

        nn.init.xavier_normal_(self.qa_outputs[0].weight)
        nn.init.xavier_normal_(self.qa_outputs[2].weight)
        
        nn.init.xavier_normal_(self.gru_aggregation.weight)
        for param in self.bigru.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

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
        train=True,
        ans_type_label=None
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
        sequence_output = graph_emb['tok'].unsqueeze(0)
         # answer type logits
        ans_type_logits = self.answer_type_classifier(self.attention(graph_out['sent']['emb'].view(1,-1,dict_params['out_feats']),
                                                                     graph_out['sent']['logits'][:,1].view(1,-1,1))
                                                      ).squeeze(1)
        loss_ans_type = loss_fn_ans_type(ans_type_logits, ans_type_label)
        span_loss = None
        start_logits = None
        end_logits = None
        if (train and ans_type_label == 0) or (not train):
            # span prediction    
            span_loss, start_logits, end_logits = self.span_prediction(sequence_output, start_positions, end_positions)
            assert not torch.isnan(start_logits).any()
            assert not torch.isnan(end_logits).any()
        
        # loss
        final_loss = self.weight_ans_type_loss + loss_ans_type
        if span_loss is not None:
            final_loss += self.weight_span_loss*span_loss
        if graph_out['sent']['loss'] is not None:
            final_loss += self.weight_sent_loss*graph_out['sent']['loss']
        if graph_out['srl']['loss'] is not None:
            final_loss += self.weight_srl_loss*graph_out['srl']['loss']
        if graph_out['ent']['loss'] is not None:
            final_loss += self.weight_ent_loss*graph_out['ent']['loss']
       
        return {'loss': final_loss, 
                'sent': graph_out['sent'], 
                'ent': graph_out['ent'],
                'srl': graph_out['srl'],
                'ans_type': {'loss': loss_ans_type, 'logits': ans_type_logits},
                'span': {'loss': span_loss, 'start_logits': start_logits, 'end_logits': end_logits}}  
    
    def attention(self, x, z):
        # x: batch_size X max_nodes X feat_dim
        # z: attention logits
        att = nn.functional.softmax(z, dim=1) # batch_size X max_nodes X 1
        output = torch.bmm(att.transpose(1,2), x)
        return output

    def graph_forward(self, graph, bert_context_emb, train):
        # create graph initial embedding #
        graph_emb = self.graph_initial_embedding(graph, bert_context_emb)
        for (k,v) in graph_emb.items():
            assert not torch.isnan(v).any()
        # graph_emb shape [num_nodes, in_feats]    
        sample_sent_nodes = self.sample_sent_nodes(graph)
        sample_srl_nodes = self.sample_srl_nodes(graph)
        sample_ent_nodes = self.sample_ent_nodes(graph)
        initial_graph_emb = graph_emb # for skip-connection
        
        # update graph embedding #
        graph_emb = self.rgcn(graph, graph_emb, bert_context_emb[0])
        
        # graph_emb shape [num_nodes, num_heads, in_feats] num_heads = 1
#         graph_emb = graph_emb.view(-1, dict_params['out_feats'])
#         # graph_emb shape [num_nodes, in_feats]

        # classify nodes #
        sent_labels = None
        ent_labels = None
        sent_emb = None
        if train:
            sent_emb = graph_emb['sent'][sample_sent_nodes]
            # add skip-connection
            query_emb = torch.cat([graph_emb['query'] for i in range(len(sample_sent_nodes))], dim=0)
            logits_sent = self.sent_classifier(torch.cat((query_emb,
                                                          graph_emb['sent'][sample_sent_nodes]), dim=1))
            assert not torch.isnan(logits_sent).any() 
            
            # contains the indexes of the srl nodes
            if len(sample_srl_nodes) == 0:
                logits_srl = None
                srl_labels = None
            else:
                query_emb = torch.cat([graph_emb['query'] for i in range(len(sample_srl_nodes))], dim=0)
                logits_srl = self.srl_classifier(torch.cat((query_emb,
                                                            graph_emb['srl'][sample_srl_nodes]), dim=1))
                # shape [num_ent_nodes, 2] 
                assert not torch.isnan(logits_srl).any()
                srl_labels = graph.nodes['srl'].data['labels'][sample_srl_nodes].to(device)
                # shape [num_sampled_srl_nodes, 1]
            
            # contains the indexes of the ent nodes
            if len(sample_ent_nodes) == 0:
                logits_ent = None
                ent_labels = None
            else:
                query_emb = torch.cat([graph_emb['query'] for i in range(len(sample_ent_nodes))], dim=0)
                logits_ent = self.ent_classifier(torch.cat((query_emb,
                                                            graph_emb['ent'][sample_ent_nodes]), dim=1))
                # shape [num_ent_nodes, 2] 
                assert not torch.isnan(logits_ent).any()
                ent_labels = graph.nodes['ent'].data['labels'][sample_ent_nodes].to(device)
                # shape [num_sampled_ent_nodes, 1]    
            sent_labels = graph.nodes['sent'].data['labels'][sample_sent_nodes].to(device)
            # shape [num_sampled_sent_nodes, 1]
            
            
        else:
            sent_emb = graph_emb['sent']
            # add skip-connection
            query_emb = torch.cat([graph_emb['query'] for i in range(graph_emb['sent'].shape[0])], dim=0)
            logits_sent = self.sent_classifier(torch.cat((query_emb, 
                                                          graph_emb['sent']), dim=1))
            assert not torch.isnan(logits_sent).any()
            query_emb = torch.cat([graph_emb['query'] for i in range(graph_emb['srl'].shape[0])], dim=0)
            logits_srl = self.srl_classifier(torch.cat((query_emb,
                                                        graph_emb['srl']), dim=1))
            # shape [num_ent_nodes, 2] 
            assert not torch.isnan(logits_srl).any()
            logits_ent = None
            ent_labels = None
            if 'ent' in graph.ntypes:
                query_emb = torch.cat([graph_emb['query'] for i in range(graph_emb['ent'].shape[0])], dim=0)
                logits_ent = self.ent_classifier(torch.cat((query_emb,
                                                            graph_emb['ent']), dim=1))
                # shape [num_ent_nodes, 2]
                assert not torch.isnan(logits_ent).any()
                ent_labels = graph.nodes['ent'].data['labels'].to(device)
                # shape [num_srl_nodes, 1]
                
            # labels
            sent_labels = graph.nodes['sent'].data['labels'].to(device)
            # shape [num_sent_nodes, 1]
            srl_labels = graph.nodes['srl'].data['labels'].to(device)
            # shape [num_sampled_srl_nodes, 1]
            
           
        # sent loss
        loss_sent = loss_fn(logits_sent, sent_labels.view(-1).long())
        probs_sent = F.softmax(logits_sent, dim=1).cpu()
        # shape [num_sent_nodes, 2]
        
        # srl loss
        loss_srl = None # not all ans are inside an srl arg
        probs_ent = torch.tensor([], device=device)
        if logits_srl is None:
            loss_srl = None
            probs_srl = None
        else:
            loss_srl = loss_fn(logits_srl, srl_labels.view(-1).long())
            probs_srl = F.softmax(logits_srl, dim=1).cpu()
            # shape [num_srl_nodes, 2]

        # ent loss
        loss_ent = None # not all ans are an entity
        probs_ent = torch.tensor([], device=device)
        if logits_ent is None:
            loss_ent = None
            probs_ent = None
        else:        
            loss_ent = loss_fn(logits_ent, ent_labels.view(-1).long())
            probs_ent = F.softmax(logits_ent, dim=1).cpu()
            # shape [num_ent_nodes, 2]

        # labels to cpu
        sent_labels = sent_labels.cpu().view(-1)
        if srl_labels is not None:
            srl_labels = srl_labels.cpu().view(-1)
        if ent_labels is not None:
            ent_labels = ent_labels.cpu().view(-1)

        return ({'sent': {'loss': loss_sent, 'probs': probs_sent, 'logits': logits_sent, 
                          'emb': sent_emb, 'lbl': sent_labels},
                'srl': {'loss': loss_srl, 'probs': probs_srl, 'lbl': srl_labels},
                'ent': {'loss': loss_ent, 'probs': probs_ent, 'lbl': ent_labels},
                },
                graph_emb)
    
    def graph_initial_embedding(self, graph, bert_context_emb):
        '''
        Inputs:
            - graph
            - bert_context_emb shape [1, #max len, 768]
        '''
        input_gru = bert_context_emb[0].view(-1, 1, dict_params['in_feats'])
        encoder_output, encoder_hidden = self.bigru(input_gru)
        encoder_output = encoder_output.view(-1, dict_params['in_feats']*2)
        graph_emb = {ntype : nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), dict_params['in_feats']))
                      for ntype in graph.ntypes if graph.number_of_nodes(ntype) > 0}
        for ntype in graph.ntypes:
            if graph.number_of_nodes(ntype) == 0:
                continue
            list_emb = []
            for (st, end) in graph.nodes[ntype].data['st_end_idx']:
                node_token_emb = encoder_output[st:end]
                left2right = node_token_emb[-1, :dict_params['in_feats']].view(-1, dict_params['in_feats'])
                right2left = node_token_emb[0, dict_params['in_feats']:].view(-1, dict_params['in_feats'])
                # concat
                concat_both_dir = torch.cat((left2right, right2left), dim=1)
                list_emb.append(concat_both_dir.squeeze(0))
            list_emb = torch.stack(list_emb, dim=0)
            graph_emb[ntype] = self.node_norm(self.gru_aggregation(list_emb))
        return graph_emb
    
    def aggregate_emb(self, encoder_output):      
        left2right = encoder_output[-1, :dict_params['in_feats']].view(-1, dict_params['in_feats'])
        right2left = encoder_output[0, dict_params['in_feats']:].view(-1, dict_params['in_feats'])
        # concat
        concat_both_dir = torch.cat((left2right, right2left), dim=1)
        # create the emb
        emb = self.gru_aggregation(concat_both_dir).squeeze(0)
        return emb 
#         return torch.mean(token_emb, dim = 0)
    
    def sample_sent_nodes(self, graph):
        list_sent_nodes = graph.nodes('sent')
        sent_nodes_labels = graph.nodes['sent'].data['labels']
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
        list_srl_nodes = graph.nodes('srl')
        srl_nodes_labels = graph.nodes['srl'].data['labels']
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
        if 'ent' not in graph.ntypes:
            return []
        list_ent_nodes = graph.nodes('ent')
        ent_nodes_labels = graph.nodes['ent'].data['labels']
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
def get_ans_type_lbl(ins):
    if ins['answer'] == 'yes':
        return torch.Tensor([1]).long().to(device)
    elif ins['answer'] == 'no':
        return torch.Tensor([2]).long().to(device)
    else:
        return torch.Tensor([0]).long().to(device)


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
#     ans_type_lbl = get_ans_type_lbl(hotpot_train[step])
#     output = model(b_graph,
#                    input_ids=input_ids,
#                    attention_mask=attention_mask,
#                    token_type_ids=token_type_ids, 
#                    start_positions=start_positions,
#                    end_positions=end_positions,
#                    ans_type_label=ans_type_lbl)
# #     total_loss = output['loss']
# #     total_loss.backward()
# #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# #     optimizer.step()
# #     scheduler.step()
# #     model.zero_grad()

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
epochs = 3

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
#                                                                num_warmup_steps = 1000, # Default value in run_glue.py
#                                                                num_training_steps = total_steps,
#                                                                num_cycles=dict_params['num_cycles'])

# %%


train_batch_size = 1


# %%
# import neptune
# neptune.init(
#     "haritz/srl-pred"
# )
# neptune.set_project('haritz/srl-pred')
# PARAMS = {"num_epoch": epochs, 
#           'lr': lr, 
#           'pretrained_weights': pretrained_weights,
#           'loss_fn': 'crossentropy', 
#           #'validation_size': len(validation_dataloader)*val_batch_size , 
#           'random_seed': random_seed,
#           'total_steps': total_steps, 
#           'training_size': len(train_dataloader)*train_batch_size, 
#           'train_batch_size': train_batch_size,
#           #'val_batch_size': val_batch_size, 
#           'scheduler': 'get_linear_schedule_with_warmup'}
# PARAMS.update(dict_params)


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
                   'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0, 
                   'srl_em': 0, 'srl_f1': 0, 'srl_prec': 0, 'srl_recall': 0,
                   'srl_recall@1': 0, 'srl_recall@3': 0, 'srl_recall@5': 0,
                   'ent_em': 0, 'ent_f1': 0, 'ent_prec': 0, 'ent_recall': 0,
                   'ent_recall@1': 0, 'ent_recall@3': 0, 'ent_recall@5': 0,
                   }
        # Evaluate data for one epoch       
        num_valid_examples = 0
        output_pred_sp = {}
        output_predictions_ans = {}
        for step, b_graph in enumerate(tqdm(self.validation_dataloader)):
            num_valid_examples += 1
            ans_type_lbl = get_ans_type_lbl(self.dataset[step])
            with torch.no_grad():
                output = self.model(b_graph,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0).to(device),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0).to(device),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0).to(device), 
                               start_positions=torch.tensor([self.list_span_idx[step][0]], device='cuda'),
                               end_positions=torch.tensor([self.list_span_idx[step][1]], device='cuda'), 
                               ans_type_label=ans_type_lbl,
                                    train=False)
            _id = self.dataset[step]['_id']   
            # Accumulate the validation loss.
            metrics['validation_loss'] += output['loss'].item()
            # Sentence evaluation
            sent_labels = output['sent']['lbl']
            prediction_sent = torch.argmax(output['sent']['probs'], dim=1)
            sp_em, sp_prec, sp_recall = self.update_sp_metrics(metrics, prediction_sent, sent_labels)
            sent_num = 0
            ## get sent titles and sents idx
            dict_ins2dict_doc2pred = self.get_oracle_dict_ins2dict_doc2pred()
            dict_sent_num2str = dict()
            for doc_idx, (doc_title, doc) in enumerate(self.dataset[step]['context']):
                if dict_ins2dict_doc2pred[step][doc_idx] == 1:
                    for i, sent in enumerate(doc):
                        dict_sent_num2str[sent_num] = {'sent': i, 'doc_title': doc_title}
                        sent_num += 1
            output_pred_sp[_id] = []
            for i, pred in enumerate(prediction_sent):
                if pred == 1:
                    output_pred_sp[_id].append([dict_sent_num2str[i]['doc_title'],
                                                dict_sent_num2str[i]['sent']])
            
            # srl
            prediction_srl = torch.argmax(output['srl']['probs'], dim=1)
            srl_labels = output['srl']['lbl']
            self.update_srl_metrics(metrics, prediction_srl, srl_labels, output['srl']['probs'][:,1])
            # ent
            if output['ent']['probs'] is not None:
                prediction_ent = torch.argmax(output['ent']['probs'], dim=1)
                ent_labels = output['ent']['lbl']
                self.update_ent_metrics(metrics, prediction_ent, ent_labels, output['ent']['probs'][:,1])
            # answer type prediction
            ans_type = torch.argmax(output['ans_type']['logits']).item()
            # answer span prediction
            golden_ans = self.dataset[step]['answer']
            predicted_ans = ""
            if ans_type == 0:
                ## wh type
                query = self.dataset[step]['question']
                wh = self.__findWHword(query)
                max_ans_len = wh_ans_len[wh]
                predicted_ans = self.__get_pred_ans_str(self.tensor_input_ids[step], output, max_ans_len)
            elif ans_type == 1:
                predicted_ans = 'yes'
            elif ans_type == 2:
                predicted_ans = 'no'
            ans_em, ans_prec, ans_recall = self.update_answer_metrics(metrics, predicted_ans, golden_ans)
            # joint
            self.update_joint_metrics(metrics, ans_em, ans_prec, ans_recall, sp_em, sp_prec, sp_recall)                
            output_predictions_ans[_id] = predicted_ans
        #N = len(self.validation_dataloader)
        N = num_valid_examples
        for k in metrics.keys():
            metrics[k] /= N
        pred_json = {'answer': output_predictions_ans, 'sp': output_pred_sp}
        return metrics, pred_json
    
    def get_answer_predictions(self, dict_ins2dict_doc2pred):
        output_pred_sp = {}
        output_predictions_ans = {}
        output_ent = {}
        output_srl = {}
        for step, b_graph in enumerate(tqdm(self.validation_dataloader)): 
            with torch.no_grad():
                ans_type_lbl = get_ans_type_lbl(self.dataset[step])
                output = self.model(b_graph,
                               input_ids=self.tensor_input_ids[step].unsqueeze(0).to(device),
                               attention_mask=self.tensor_attention_masks[step].unsqueeze(0).to(device),
                               token_type_ids=self.tensor_token_type_ids[step].unsqueeze(0).to(device), 
                               start_positions=torch.tensor([self.list_span_idx[step][0]], device='cuda'),
                               end_positions=torch.tensor([self.list_span_idx[step][1]], device='cuda'), 
                               ans_type_label=ans_type_lbl,
                                train=False)
            _id = self.dataset[step]['_id']
            ans_type = torch.argmax(output['ans_type']['logits']).item()
            # answer span prediction
            golden_ans = self.dataset[step]['answer']
            predicted_ans = ""
            if ans_type == 0:
                ## wh type
                query = self.dataset[step]['question']
                wh = self.__findWHword(query)
                max_ans_len = wh_ans_len[wh]
                predicted_ans = self.__get_pred_ans_str(self.tensor_input_ids[step], output, max_ans_len)
            elif ans_type == 1:
                predicted_ans = 'yes'
            elif ans_type == 2:
                predicted_ans = 'no'
            output_predictions_ans[_id] = predicted_ans
            # ent
#             ent = self.__get_ent_str(b_graph, self.tensor_input_ids[step], output)
#             output_ent[_id] = ent
#             # srl
#             srl = self.__get_srl_str(b_graph, self.tensor_input_ids[step], output)
#             output_srl[_id] = srl
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
    
    def get_oracle_dict_ins2dict_doc2pred(self):
        dict_ins2dict_doc2pred = dict()
        for ins_idx, ins in enumerate(hotpot_dev):
            set_golden_docs = set([title for title, _ in ins['supporting_facts']])
            dict_doc2pred = dict()
            for doc_idx, (title, doc) in enumerate(ins['context']):
                if title in set_golden_docs:
                    dict_doc2pred[doc_idx] = 1
                else:
                    dict_doc2pred[doc_idx] = 0
            dict_ins2dict_doc2pred[ins_idx] = dict_doc2pred
        return dict_ins2dict_doc2pred
    
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
# model = HGNModel.from_pretrained('../../models/ans_type_pred_3_epochs')
# model.cuda()

# %%
# validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
#                         dev_tensor_input_ids, dev_tensor_attention_masks, 
#                         dev_tensor_token_type_ids,
#                         dev_list_span_idx)
# metrics, pred_json_oracle = validation.do_validation()

# %%
# metrics

# %%
# with open('../../preds_ans_type_pred_3_epochs_oracle.json', 'w+') as f:
#     json.dump(pred_json_oracle, f)

# %%
# # %%


# # %%


# # %%
# list(preds['answer'].items())[:50]

# # %%


# # %%
# validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
#                         dev_tensor_input_ids, dev_tensor_attention_masks, 
#                         dev_tensor_token_type_ids,
#                         dev_list_span_idx)
# metrics = validation.do_validation()

# # %%
# metrics

# %%
# validation = Validation(model, hotpot_dev, dev_list_graphs[:10], tokenizer,
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
model_path = 'models/hsgn_2021_fix'

best_eval_em = 0
# Measure the total training time for the whole run.
total_t0 = time.time()
# with neptune.create_experiment(name="2021", params=PARAMS, upload_source_files=['src/models/GAT_Hierar_Tok_Node_Aggr.py']):
#     neptune.set_property('server', 'irgpu11')
#     neptune.set_property('training_set_path', training_path)
#     neptune.set_property('dev_set_path', dev_path)

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
        b_graph = list_graphs[idx]
        # neptune.log_metric('step', step)
        # forward
        input_ids=tensor_input_ids[idx].unsqueeze(0).to(device)
        attention_mask=tensor_attention_masks[idx].unsqueeze(0).to(device)
        token_type_ids=tensor_token_type_ids[idx].unsqueeze(0).to(device) 
        start_positions=torch.tensor([list_span_idx[idx][0]], device='cuda')
        end_positions=torch.tensor([list_span_idx[idx][1]], device='cuda')
        ans_type_lbl = get_ans_type_lbl(hotpot_train[idx])
        output = model(b_graph,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids, 
                        start_positions=start_positions,
                        end_positions=end_positions,
                        ans_type_label=ans_type_lbl)
        
        total_loss = output['loss'] / dict_params['accumulation_steps']
        assert not torch.isnan(total_loss)
        sent_loss = output['sent']['loss']
        ent_loss = output['ent']['loss']
        srl_loss = output['srl']['loss']
        span_loss = output['span']['loss']
        ans_type_loss = output['ans_type']['loss']
        # neptune
        # neptune.log_metric("total_loss", total_loss.detach().item())
        # if sent_loss is not None:
        #     neptune.log_metric("sent_loss", sent_loss.detach().item())
        # if srl_loss is not None:
        #     neptune.log_metric("srl_loss", srl_loss.detach().item())
        # if ent_loss is not None:    
        #     neptune.log_metric("ent_loss", ent_loss.detach().item())
        # if span_loss is not None:
        #     neptune.log_metric("span_loss", span_loss.detach().item())
        # if ans_type_loss is not None:
        #     neptune.log_metric("ans_type_loss", ans_type_loss.detach().item())

        # backpropagation
        total_loss.backward()
        if (step + 1) % dict_params['accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            if epoch_i == 0 and (step +1) == 10000:
                #############################
                ######### Validation ########
                #############################
                validation = Validation(model, hotpot_dev, dev_list_graphs, tokenizer,
                                        dev_tensor_input_ids, dev_tensor_attention_masks, 
                                        dev_tensor_token_type_ids,
                                        dev_list_span_idx)
                metrics, pred_json = validation.do_validation()
                model.train()
                # record_eval_metric(neptune, metrics)

                curr_em = metrics['ans_em']
                if  curr_em > best_eval_em:
                    best_eval_em = curr_em
                    model.save_pretrained(model_path) 
                    with open(os.path.join(model_path, 'output.json'), 'w+') as f:
                        json.dump(pred_json, f)
            if epoch_i == 2 and (step +1) % 10000 == 0:
                model_path_step = model_path + "/epoch3/step_" + str(step)
                os.mkdir(model_path_step)
                model.save_pretrained(model_path_step)
        total_train_loss += total_loss.detach().item()

        # free-up gpu memory
        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        token_type_ids = token_type_ids.to('cpu')
        start_positions = start_positions.to('cpu')
        end_positions = end_positions.to('cpu')
        
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
    metrics, pred_json = validation.do_validation()
    model.train()
    # record_eval_metric(neptune, metrics)

    curr_em = metrics['ans_em']
    if  curr_em > best_eval_em:
        best_eval_em = curr_em
        model.save_pretrained(model_path) 
        with open(os.path.join(model_path, 'output.json'), 'w+') as f:
            json.dump(pred_json, f)

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
#     zipdir(model_path, os.path.join(model_path, 'checkpoint.zip'))
#     # upload the model to neptune
#     neptune.send_artifact(os.path.join(model_path, 'checkpoint.zip'))
