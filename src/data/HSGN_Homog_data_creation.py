# %%
import re

from fuzzywuzzy import fuzz
import json
import pickle
import warnings
import os
from os import listdir
from os.path import isfile, join
from itertools import product
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import *

import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch.conv import GATConv
from dgl.data.utils import save_graphs

os.environ['DGLBACKEND'] = 'pytorch'
warnings.filterwarnings(action='once')

# %%
import random
random_seed = 2020
# Set the seed value all over the place to make this reproducible.
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# %%
data_path = '/workspace/ml-workspace/thesis_git/thesis/data/'
hotpotqa_path = 'external/'
intermediate_train_data_path = 'interim/training/'
intermediate_dev_data_path = 'interim/dev/'
with open(os.path.join(data_path, hotpotqa_path, "hotpot_train_v1.1.json"), "r") as f:
    hotpot_train = json.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "list_hotpot_ner_no_coref_train.p"), "rb") as f:
    list_hotpot_train_ner = pickle.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "dict_ins_doc_sent_srl_triples.json"), 'r') as f:
    dict_ins_doc_sent_srl_triples = json.load(f)
with open(os.path.join(data_path, hotpotqa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
    hotpot_dev = json.load(f)
with open(os.path.join(data_path, intermediate_dev_data_path, "list_hotpot_ner_no_coref_dev.p"), "rb") as f:
    list_hotpot_dev_ner = pickle.load(f)
with open(os.path.join(data_path, intermediate_dev_data_path, "dict_ins_doc_sent_srl_triples_dev.json"), 'r') as f:
    dict_ins_doc_sent_srl_triples_dev = json.load(f)

# %%
# Some constants
device = 'cuda'
pretrained_weights = 'bert-base-cased'
MAX_LEN = 512
node_type2idx = {'doc': 0, 'sent': 1, 'srl': 2, 'ent': 3, 'token': 4, 'query': 5}

# Graph Creation

## Auxiliary functions
# %%
def find_sublist_idx(x: list, y: list) -> int:
    '''
    Return the first index of the sublist in the list
    Input:
        - x: list
        - y: sublist
    Returns:
        - index of the occurence of the sublist in the list
    '''
    occ = [i for i, a in enumerate(x) if a == y[0]]
    
    for b in occ:
        # check if the full sublist is in the list
        if x[b:b+len(y)] == y:
            return b
        if len(occ)-1 ==  occ.index(b):
            # check all possible sublist candidates but not found the full sublist
            # return the first occurrence. Be careful, it can lead to wrong results
            # but in 99% of the cases should be fine.
            # If we reach this case, it is becase the SRL model skipped some token
            # i.e. B-arg0, I-arg0, O, I-arg0,...
            return occ[0]
    raise Exception("Sublist not in list")

def test_find_sublist_idx():
    x = [0,1,2,3,4,5,6,7]
    y = [3,4,5]
    assert find_sublist_idx(x, y) == 3
    x = [0,1,2,3,4,2,3,4,5]
    y = [3,4,5]
    assert find_sublist_idx(x, y) == 6
    x = [0,1,2,3,4,2,3,4,5]
    y = [10, 11]
    try:
        assert find_sublist_idx(x, y)
    except:
        assert True

def ans_type(ans):
    if ans == 'yes':
        return 1
    elif ans == 'no':
        return 2
    else:
        return 0
# %% [markdown]
# # Graph Creation
#
#
# ### Code
#
# # Description
# ## Nodes:
#     doc, sent, srl, ent, token
# ## Edges
#     * Hierarchical
#         * doc -> sent # lbl: [SENT2DOC]
#         * doc <- sent # lbl: [DOC2SENT]
#         * sent -> srl # lbl: [SENT2SRL]
#         * sent <- srl # lbl: [SRL2SENT]
#         * srl -> ent  # lbl: [SRL2ENT]
#         * srl <- ent  # lbl: [ENT2SRL]
#     * SameLevel
#         * self doc <-> doc # lbl: [DOC2DOC_SELF] self edge only
#         * sent <-> sent    # lbl: [SENT2SENT] all sent in a doc are fully connected # comment from Prof. only sequentially to save time and fully connected may not add anything
#         * srl <-> srl      # lbl: [SRL2SRL] all srl of a triple are fully connected
#         * self ENT <-> ENT # lbl: [ENT2ENT_SELF]
#         * self tok <-> tok # lbl: [TOK2TOK_SELF]
#     - To token emb
#         * doc -> tok  # lbl: [DOC2TOK]
#         * doc <- tok  # lbl: [TOK2DOC]
#         * sent -> tok # lbl: [SENT2TOK]
#         * sent <- tok # lbl: [TOK2SENT]
#         * srl -> tok  # lbl: [SRL2TOK]
#         * srl <- tok  # lbl: [TOK2SRL]
#         * ent -> tok  # lbl: [ENT2TOK]
#         * ent <- tok  # lbl: [TOK2ENT]
#
#     - multihop

# %%
class Dataset():
    def __init__(self, dataset = None, list_hotpot_ner = None, dict_ins_doc_sent_srl_triples = None,
                 batch_size = None, max_len = 512):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights,
                                                       do_basic_tokenize=True,
                                                       clean_text=False)
        self.dataset = dataset
        self.list_hotpot_ner = list_hotpot_ner
        self.dict_ins_doc_sent_srl_triples = dict_ins_doc_sent_srl_triples
        self.batch_size = batch_size
        self.max_len = max_len
        
    def create_dataloader(self):
        list_context, list_dict_idx = self.encode_all_sentences()
        list_graphs = []
        list_span_idx = []
        for ins_idx, hotpot_instance in enumerate(tqdm(self.dataset)):
            list_entities = self.list_hotpot_ner[ins_idx]
            dict_idx = list_dict_idx[ins_idx]
            g, span_idx = self.create_graph(hotpot_instance, list_entities, dict_idx, list_context[ins_idx]['input_ids'], ins_idx)
            self.build_tests(ins_idx, g, span_idx, hotpot_instance['answer'])
            list_graphs.append(g)
            list_span_idx.append(span_idx)
        return list_graphs, list_context, list_span_idx

    def build_tests(self, ins_idx, g, span_idx, ans):
        #print(ins_idx, span_idx)
        if ans == 'yes' or ans == 'no':
            assert span_idx == (-1, -1)
        assert span_idx[0] < self.max_len
        assert span_idx[1] < self.max_len
            
    def collate(self, graphs):
        # The input `samples` is a list of pairs
        #  (graph, input_ids, attention_masks, token_types, label).
        batched_graph = dgl.batch(graphs)
        return batched_graph
  
    def encode_all_sentences(self):
        encoding, list_query_idx, list_instance2idx = self.batch_encoding(self.dataset)
        list_context = []
        list_dict_idx = []
        for i, dict_doc_metadata in enumerate(list_instance2idx):
            #dict_doc_metadata keys: list_list_sent_encoding_idx and list_golden_doc_idx
            context, dict_idx = self.create_context(encoding, list_query_idx[i], dict_doc_metadata)
            list_dict_idx.append(dict_idx)
            list_context.append(context)
        return list_context, list_dict_idx
    
    def batch_encoding(self, dataset):
        list_instance2idx = []
        list_query_idx = []
        list_instance_list_doc_idx = []
        idx = 0
        list_sent = []
        for hotpot_idx, hotpot_instance in enumerate(self.dataset):
            # query
            list_query_idx.append(idx)
            list_sent.append(hotpot_instance['question'])
            idx += 1
            # doc
            list_doc_list_sent_idx = []
            list_golden_doc_idx = []
            set_supporting_doc_titles = self.get_sup_doc_titles(hotpot_instance)
            # for each doc
            for doc_idx, (doc_title, doc) in enumerate(hotpot_instance['context']):
                if doc_title not in set_supporting_doc_titles:
                    continue
                list_golden_doc_idx.append(doc_idx)
                doc_init = idx
                # for each sent
                list_sent_idx = []
                for sent_idx, sent in enumerate(doc):
                    list_sent.append(sent)
                    list_sent_idx.append(idx)
                    idx += 1
                list_doc_list_sent_idx.append(list_sent_idx)
                
            list_instance2idx.append({"list_list_sent_encoding_idx": list_doc_list_sent_idx, 
                                      "list_golden_doc_idx": list_golden_doc_idx})
        encoding = self.tokenizer.batch_encode_plus(list_sent,
                                                    add_special_tokens=False,
                                                    truncation=True,
                                                    pad_to_max_length=False,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True,
                                                    )
        return encoding, list_query_idx, list_instance2idx

    def create_context(self, encoding: dict, query_idx: int, dict_doc_metadata: dict):
        '''
        Inputs:
            - encoding: dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
            - query_idx: indexes of the queries in encoding
            - list_list_sent_idx: list of list of idx (in encoding) of setences. The first dimension represents docs.
                eg: [[70, 71, 72], [73, 74, 75, 76]], so 2 docs (3 sents, 4 sents)
        Return:
            - context: input_ids of question and list of documents (with special tokens)
            - dict_idx: dict with the idx of the query, doc, and sentences
                eg: {'q': (1, 21),
                     'list_docs': [{'idx_doc': (22, 136),
                       'idx_sent': [(22, 41), (42, 89), (90, 136)]},
                      {'idx_doc': (137, 275),
                       'idx_sent': [(137, 182), (183, 220), (221, 235), (236, 275)]}]}
        '''
        context_input_ids = []
        context_token_type_ids = []
        context_attention_mask = []
        context_input_ids.append(101) # CLS
        q_start = len(context_input_ids)
        context_input_ids.extend(encoding['input_ids'][query_idx]) # QUERY
        q_end = len(context_input_ids)
        context_input_ids.append(102) # SEP (between query and sentences)
        # question tokens token type = 0
        context_token_type_ids = [0] * len(context_input_ids)
        dict_idx = {"q_token_st_end_idx": (q_start, q_end)}
        list_docs = []
        for list_sent_encoding_idx in dict_doc_metadata['list_list_sent_encoding_idx']:
            list_sent = []
            for sent_encoding_idx in list_sent_encoding_idx:
                s_start = len(context_input_ids)
                context_input_ids.extend(encoding['input_ids'][sent_encoding_idx])
                s_end = len(context_input_ids)
                list_sent.append((s_start, s_end))
                if s_end > self.max_len - 1:
                    s_end = self.max_len - 1
                    break
            
            dict_docs = {"doc_token_st_end_idx":(list_sent[0][0], list_sent[-1][-1]),
                         "list_sent_token_st_end_idx": list_sent
                        }
            list_docs.append(dict_docs)
            if s_end == self.max_len - 1:
                break
        dict_idx["list_docs_encoding_metadata"] = list_docs
        dict_idx['list_golden_doc_idx'] = dict_doc_metadata['list_golden_doc_idx']
        context_input_ids.append(102) # SEP (last one)
        # attention 1 to all valid tokens
        context_attention_mask = [1] * len(context_input_ids)
        # token type 1 to evid doc
        context_token_type_ids.extend([1] * (len(context_input_ids)-len(context_token_type_ids)))
        if len(context_input_ids) < self.max_len:
            context_input_ids.extend([0]*(self.max_len-len(context_input_ids))) # 0 = [PAD]
            context_attention_mask.extend([0] * (self.max_len-len(context_attention_mask)))
            context_token_type_ids.extend([0] * (self.max_len-len(context_token_type_ids)))
        else:
            context_input_ids = context_input_ids[:self.max_len]
            context_attention_mask = context_attention_mask[:self.max_len]
            context_token_type_ids = context_token_type_ids[:self.max_len]
        return ({'input_ids': context_input_ids,
                 'attention_mask': context_attention_mask,
                 'token_type_ids': context_token_type_ids}, dict_idx)

    def create_graph(self, hotpot_instance, list_entities, dict_idx, context, ins_idx):
        node_idx = 0
        
        # metadata of the graph
        list_edges = []
        list_context_idx = []
        list_st_end_idx = []
        list_node_type = []
        list_lbl = []
        
        # the first max_len nodes are for the tokens
        for i in range(self.max_len):
            list_context_idx.append(ins_idx)
            list_st_end_idx.append((i, i + 1))
            list_node_type.append(node_type2idx['token'])
            list_lbl.append(0)
            list_edges.append((i, i))  # lbl: [TOK2TOK_SELF]
            node_idx += 1
        
        ans_str = hotpot_instance['answer']
        ans_encoded = self.tokenizer.encode(ans_str, add_special_tokens=False)
        ans_detokenized = self.tokenizer.decode(ans_encoded)
        yn_ans = ans_str == 'yes' or ans_str == 'no'
        ans_st_idx = -1
        ans_end_idx = -1
        list_doc_nodes = []
        dict_srl_metadata = dict() # needed to create srl common entity edges
        # for each doc
        dict_sent_node2metadata = dict()
        for idx, doc_metadata in enumerate(dict_idx['list_docs_encoding_metadata']):
            # doc node
            current_doc_node = node_idx
            node_idx += 1
            list_doc_nodes.append(current_doc_node)
            # metadata doc
            list_context_idx.append(ins_idx)
            list_st_end_idx.append(doc_metadata['doc_token_st_end_idx'])
            list_node_type.append( node_type2idx['doc'])
            list_lbl.append(1)
            # end metada doc
            # add edges to its tokens
            (st, end) = doc_metadata['doc_token_st_end_idx']
            for tok in range(st, end):
                list_edges.append((current_doc_node, tok))  # lbl: [DOC2TOK]
                list_edges.append((tok, current_doc_node))  # lbl: [TOK2DOC]
            doc_idx = dict_idx['list_golden_doc_idx'][idx]
            #get supp sentences
            doc_title = hotpot_instance['context'][doc_idx][0]
    #         print("doc title", doc_title)
            dict_supporting_docs = self.__get_dict_supporting_doc_idx2sent_idx(hotpot_instance['supporting_facts'])
            # to fully connect all sent nodes
            list_sent_nodes = []
            ########################################################
            # for each sentence
            for sent_idx, (sent_st, sent_end) in enumerate(doc_metadata['list_sent_token_st_end_idx']):
                # hotpotqa contains some empty sentences
                if sent_st == sent_end:
                    continue
                # sent node
                current_sent_node = node_idx
                node_idx += 1
                list_sent_nodes.append(current_sent_node)
                # metadata sent
                list_context_idx.append(ins_idx)
                list_st_end_idx.append((sent_st, sent_end))
                list_node_type.append( node_type2idx['sent'])
                sent_lbl = sent_idx in dict_supporting_docs[doc_title]
                list_lbl.append(sent_lbl)
                # end metada sent
                
                for tok in range(sent_st, sent_end):
                    list_edges.append((current_sent_node, tok))  # lbl: [SENT2TOK]
                    list_edges.append((tok, current_sent_node))  # lbl: [TOK2SENT]
                
                if sent_lbl and not yn_ans:
                    try:
                        ans_st_idx = find_sublist_idx(context[sent_st:sent_end], ans_encoded) + sent_st
                        ans_end_idx = min(ans_st_idx + len(ans_encoded), self.max_len-1)
                    except:
                        pass
                dict_sent_node2metadata[current_sent_node] = {'doc_idx': doc_idx, 'sent_idx': sent_idx}
                sent_str = self.tokenizer.decode(context[sent_st:sent_end])
#                 print("### sent", lbl, sent_str)
                ########################################################################
                # for each SRL triple
                for key, triple_dict in self.dict_ins_doc_sent_srl_triples[str(ins_idx)][str(doc_idx)][str(sent_idx)].items():
                    list_arg_nodes = [] # list of all arguments (node idx) in the triple
                    for arg_type, arg_str in triple_dict.items():
                        try:
                            arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                            arg_str = self.tokenizer.decode(arg_encoded)
                            # find location of the entity in the context (inputs ids)
                            # +sent_st 'cuz I need the index in the full context
                            # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                            st_tok_idx = find_sublist_idx(context[sent_st:sent_end], arg_encoded) + sent_st
                            end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                            # metadata ent
                            list_context_idx.append(ins_idx)
                            list_st_end_idx.append((st_tok_idx, end_tok_idx))
                            list_node_type.append(node_type2idx['srl'])
                            srl_lbl = self.srl_with_ans(arg_str, ans_detokenized, sent_lbl)
                            list_lbl.append(srl_lbl)
                            # ent node
                            current_srl_node = node_idx
                            node_idx += 1
                            list_arg_nodes.append(current_srl_node)
                            
                            # srl <-> token
                            for tok in range(st_tok_idx, end_tok_idx):
                                list_edges.append((current_srl_node, tok)) # lbl: [SRL2TOK]
                                list_edges.append((tok, current_srl_node))  # lbl: [TOK2SRL]
                            #######################################################
                            # for each entity
                            list_srl_ent = []
                            for ent in list_entities[doc_idx][sent_idx]:
                                try:
                                    # compute this beforehand ################################ here!!!!!!!!!!!!!!!!!!!
                                    ent_encoded = self.tokenizer.encode(ent, add_special_tokens=False)
                                    ent = self.tokenizer.decode(ent_encoded)
                                    if ent not in arg_str:
                                        continue
                                    list_srl_ent.append(ent)
                                    # find location of the entity in the context (inputs ids)
                                    # +sent_st 'cuz I need the index in the full context
                                    # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                                    st_tok_idx = find_sublist_idx(context[sent_st:sent_end], ent_encoded) + sent_st
                                    end_tok_idx = min(st_tok_idx + len(ent_encoded), self.max_len)
                                    # metadata ent
                                    list_context_idx.append(ins_idx)
                                    list_st_end_idx.append((st_tok_idx, end_tok_idx))
                                    list_node_type.append(node_type2idx['ent'])
                                    lbl = (fuzz.token_set_ratio(ans_detokenized, ent) >= 90)
                                    list_lbl.append(lbl)
                                    if (not srl_lbl) and lbl:
                                        srl_lbl = True
                                        list_lbl[-1] = srl_lbl
                                    # ent node
                                    current_ent_node = node_idx
                                    node_idx += 1
                                    # add edges (srl_arg -> ent and ent -> srl_arg)
                                    list_edges.append((current_srl_node, current_ent_node))  # lbl: [SRL2ENT]
                                    list_edges.append((current_ent_node, current_srl_node))  # lbl: [ENT2SRL]
                                    # self edge ent
                                    list_edges.append((current_ent_node, current_ent_node))  # lbl: [ENT2ENT_SELF]
                                    # srl <-> token
                                    for tok in range(st_tok_idx, end_tok_idx):
                                        list_edges.append((current_ent_node, tok))  # lbl: [ENT2TOK]
                                        list_edges.append((tok, current_ent_node))  # lbl: [TOK2ENT]
            #                         print("#### ent ", lbl, self.tokenizer.decode(context[st_tok_idx:end_tok_idx]))
                                except:
            #                         print(ent)
            #                         print("### sent", sent_str)
                                    pass
                        
                            dict_srl_metadata[current_srl_node] = list_srl_ent
                        except:
                             pass
                    # lbl: [SRL2SRL] bidirectional and self-node
                    list_edges.extend([(arg1, arg2) for arg1 in list_arg_nodes
                                                    for arg2 in list_arg_nodes])
                    
                    # srl_arg -> sent # lbl: [SRL2SENT]
                    list_edges.extend([(arg, current_sent_node) for arg in list_arg_nodes])        
                    # sent -> srl_arg # lbl: [SENT2SRL]
                    list_edges.extend([(current_sent_node, arg) for arg in list_arg_nodes])
                # sent -> doc # lbl: [SENT2DOC]
                list_edges.append((current_sent_node, current_doc_node))
                # doc -> sent # lbl: [DOC2SENT]
                list_edges.append((current_doc_node, current_sent_node))
            # fully connected sent nodes # lbl: [SENT2SENT] 
            list_edges.extend([(u, v) for u in list_sent_nodes for v in list_sent_nodes])
        # self-edges to the docs
        list_edges.extend([(u, u) for u in list_doc_nodes])  # lbl: [DOC2DOC_SELF]
        # fully connected doc nodes    
        # list_edges.extend([(u, v) for u in list_doc_nodes for v in list_doc_nodes])
        ############# NER edges ##################
        list_ent_edges = self.create_common_entity_edges(dict_sent_node2metadata, list_entities)
        list_edges.extend(list_ent_edges)
        list_srl_ent_edges = self.create_common_entity_edges_srl_lvl(dict_srl_metadata)
        list_edges.extend(list_srl_ent_edges)
        ############ END Entity Edges ############
        
        ############ Query node ################
        # for yes/no answers
        q_node = node_idx
        node_idx += 1
        list_context_idx.append(ins_idx)
        (st, end) = doc_metadata['doc_token_st_end_idx']
        list_st_end_idx.append((0, end))  # the initial embedding will be the agg from token 0 to the last token
        list_node_type.append(node_type2idx['query'])
        lbl = ans_type(ans_str)
        list_lbl.append(lbl)
        list_edges.extend([(u, q_node) for u in range(self.max_len, node_idx)])
        ############ END Query node ################
        # make the graph
        graph = dgl.DGLGraph()
        graph.add_nodes(node_idx)
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*list_edges))
        # dgl is directional
        graph.add_edges(src, dst)
        # add node metadata to the graph
        graph.ndata['node_type'] = np.array(list_node_type).reshape(-1,1)
        graph.ndata['st_end_idx'] =  np.array(list_st_end_idx)
        graph.ndata['list_context_idx'] = np.array(list_context_idx).reshape(-1,1)
        graph.ndata['labels'] = np.array(list_lbl).reshape(-1,1)
        return graph, (ans_st_idx, ans_end_idx)

    
    def srl_with_ans(self, srl_arg: str, ans:str, in_supp_sent: bool) -> bool:
        return (in_supp_sent and ans != "yes" and ans != "no" and
                ((ans in srl_arg) or (ans[:-1] in srl_arg) or (fuzz.token_set_ratio(srl_arg, ans) >= 90)))
        
        
    def __get_dict_supporting_doc_idx2sent_idx(self, list_supporting_docs: list) -> dict:
        '''
        Returns dict: int (doc_idx) -> int (supporting sentence index)
        '''
        dict_doctitle2sentidx = {}
        for doc_idx, (title, sent_idx) in enumerate(list_supporting_docs):
            if title in dict_doctitle2sentidx:
                dict_doctitle2sentidx[title].append(sent_idx)
            else:
                dict_doctitle2sentidx[title] = [sent_idx]

        return dict_doctitle2sentidx
    
    def __get_dict_supporting_doc_title2sent_idx(self, list_supporting_docs: list) -> dict:
        '''
        Returns dict: str (title) -> int (supporting sentence index)
        '''
        dict_doctitle2sentidx = {}
        for (title, idx) in list_supporting_docs:
            if title in dict_doctitle2sentidx:
                dict_doctitle2sentidx[title].append(idx)
            else:
                dict_doctitle2sentidx[title] = [idx]
        return dict_doctitle2sentidx
    
    def get_sup_doc_titles(self, hotpot_instance):
        list_supporting_docs = hotpot_instance['supporting_facts']
        dict_supporting_doc_title2sent_idx = self.__get_dict_supporting_doc_title2sent_idx(list_supporting_docs)
        set_supporting_doc_titles = set(dict_supporting_doc_title2sent_idx.keys())
        return set_supporting_doc_titles
    
    def create_common_entity_edges(self, dict_sent_node2metadata: dict, list_hotpot_instance_ner: list) -> list:
        '''
        Input: dict {node_idx: {'doc_idx': doc_idx, 'sent_idx': sent_idx}}
        Output: [(node_i, node_j)]
        Check all sentences in a hotpot instance with common named entities
        Cost: O(n^2)
        '''
        list_ent_edges = []
        for (node_i, dict_sent_i) in dict_sent_node2metadata.items():
            for (node_j, dict_sent_j) in dict_sent_node2metadata.items():
                if node_i != node_j:
                    doc_i = dict_sent_i['doc_idx']
                    sent_i = dict_sent_i['sent_idx']
                    doc_j = dict_sent_j['doc_idx']
                    sent_j = dict_sent_j['sent_idx']
                    try:
                        if self.intersection(list_hotpot_instance_ner[doc_i][sent_i], 
                                             list_hotpot_instance_ner[doc_j][sent_j]):
                            list_ent_edges.append((node_i, node_j))
                    except:
                        pass
        return list_ent_edges
    
    def create_common_entity_edges_srl_lvl(self, dict_srl_node2list_ent: dict):
        list_edges = []

        # create a dict of the shape entity -> list[SRL NODE], i.e. group slr nodes by entity
        dict_ent2list_srl = dict()
        # O(#srl nodes)
        for srl_node, list_ent in dict_srl_node2list_ent.items():
            for ent in list_ent:
                if ent in dict_ent2list_srl:
                    dict_ent2list_srl[ent].append(srl_node)
                else:
                    dict_ent2list_srl[ent] = [srl_node]
        
        # create edges O(#entities)
        for e, list_srl in dict_ent2list_srl.items():
            # this list is bidirectional
            list_edges.extend([(u, v) for u in list_srl for v in list_srl if u != v])
        return list_edges

    def intersection(self, l1: list, l2: list) -> set:
        return set(l1).intersection(l2)



# %%
train_dataset = Dataset(hotpot_train, list_hotpot_train_ner, dict_ins_doc_sent_srl_triples, 1)
list_graphs, list_context, list_span_idx = train_dataset.create_dataloader()

list_input_ids = [context['input_ids'] for context in list_context]
list_token_type_ids = [context['token_type_ids'] for context in list_context]
list_attention_masks = [context['attention_mask'] for context in list_context]

tensor_input_ids = torch.tensor(list_input_ids)
tensor_token_type_ids = torch.tensor(list_token_type_ids)
tensor_attention_masks = torch.tensor(list_attention_masks)

training_path = "/workspace/ml-workspace/thesis_git/thesis/data/processed/training/homog_20200804/"
for i, g in enumerate(list_graphs):
    with open( training_path + "graphs/graph"+str(i)+".bin", "wb" ) as f:
        pickle.dump(g, f)

torch.save(tensor_input_ids, training_path + 'tensor_input_ids.p')
torch.save(tensor_token_type_ids, training_path + 'tensor_token_type_ids.p')
torch.save(tensor_attention_masks, training_path + 'tensor_attention_masks.p')

with open(training_path + 'list_span_idx.p', 'wb') as f:
    pickle.dump(list_span_idx, f)

# %%
list_graphs[1]

# %%
dev_dataset = Dataset(hotpot_dev, list_hotpot_dev_ner, dict_ins_doc_sent_srl_triples_dev, 1)
list_graphs, list_g_metadata, list_context, list_span_idx = dev_dataset.create_dataloader()
list_input_ids = [context['input_ids'] for context in list_context]
list_token_type_ids = [context['token_type_ids'] for context in list_context]
list_attention_masks = [context['attention_mask'] for context in list_context]
tensor_input_ids = torch.tensor(list_input_ids)
tensor_token_type_ids = torch.tensor(list_token_type_ids)
tensor_attention_masks = torch.tensor(list_attention_masks)

# %%
dev_path = "/workspace/ml-workspace/thesis_git/thesis/data/processed/dev/homog_20200804/"
training_path = "/workspace/ml-workspace/thesis_git/thesis/data/processed/training/homog_20200804/"

for i, g in enumerate(tqdm(list_graphs)):
    with open( dev_path + "graphs/graph"+str(i)+".bin", "wb" ) as f:
        pickle.dump(g, f)
    with open( dev_path + "metadata/metadata"+str(i)+".bin", "wb" ) as f:
        pickle.dump(list_g_metadata[i], f)
    # separate the metadata from the graph to store it (do not add metadata in the first place)
print("Saving tensors")
torch.save(tensor_input_ids, '/workspace/ml-workspace/datasets/hotpotqa/dev/hetero_hsgn/tensor_input_ids.p')
torch.save(tensor_token_type_ids, '/workspace/ml-workspace/datasets/hotpotqa/dev/hetero_hsgn/tensor_token_type_ids.p')
torch.save(tensor_attention_masks, '/workspace/ml-workspace/datasets/hotpotqa/dev/hetero_hsgn/tensor_attention_masks.p')
print("Saving list span idx")
with open('/workspace/ml-workspace/datasets/hotpotqa/dev/hetero_hsgn/list_span_idx.p', 'wb') as f:
    pickle.dump(list_span_idx, f)
