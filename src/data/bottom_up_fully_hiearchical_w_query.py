# +
import re
from transformers import BertTokenizer
import transformers.data.metrics.squad_metrics as squad_metrics
from fuzzywuzzy import fuzz
import json
import pickle
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from transformers import *
from os import listdir
from os.path import isfile, join
from itertools import product
from dgl.data.utils import load_graphs
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings(action='once')
import os
os.environ['DGLBACKEND'] = 'pytorch'

import random
random_seed = 2020
# Set the seed value all over the place to make this reproducible.
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# -
device = 'cuda'
pretrained_weights = 'bert-base-cased'

# +
data_path = '/workspace/ml-workspace/thesis_git/HSGN/data/'
hotpotqa_path = 'external/'
intermediate_train_data_path = 'interim/training/'
intermediate_dev_data_path = 'interim/dev/'

# training data
with open(os.path.join(data_path, hotpotqa_path, "hotpot_train_v1.1.json"), "r") as f:
    hotpot_train = json.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "list_hotpot_ner_no_coref_train.p"), "rb") as f:
    list_hotpot_train_ner = pickle.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "dict_ins_doc_sent_srl_triples.json"), 'r') as f:
    dict_ins_doc_sent_srl_triples = json.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "dict_ins_query_srl_triples.json"), "r") as f:
    dict_ins_query_srl_triples_training = json.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "list_ent_query_training.p"), "rb") as f:
    list_ent_query_training = pickle.load(f)
print("Training data loaded")


# -

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


# # Graph Construction

def ans_type(ans):
    if ans == 'yes':
        return 1
    elif ans == 'no':
        return 2
    else:
        return 0


# +
MAX_LEN = 512
node_type2idx = {'doc': 0, 'sent': 1, 'srl': 2, 'ent': 3, 'token': 4, 'query': 5}
class Dataset():
    def __init__(self, dataset = None, list_hotpot_ner = None, dict_ins_doc_sent_srl_triples = None,
                 dict_ins_query_triples = None, list_entities_query = None, batch_size = None, max_len = 512):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights,
                                                       do_basic_tokenize=True,
                                                       clean_text=False)
        self.dataset = dataset
        self.list_hotpot_ner = list_hotpot_ner
        self.dict_ins_doc_sent_srl_triples = dict_ins_doc_sent_srl_triples
        self.dict_ins_query_triples = dict_ins_query_triples
        self.list_entities_query = list_entities_query
        self.batch_size = batch_size
        self.max_len = max_len
        
    def create_dataloader(self):
        list_context, list_dict_idx = self.encode_all_sentences()
        list_graphs = []
        list_span_idx = []
        list_g_metadata = []
        list_list_srl_edges_metadata = []
        list_list_ent2ent_metadata = []
        for ins_idx, hotpot_instance in enumerate(tqdm(self.dataset)):
            list_entities = self.list_hotpot_ner[ins_idx]
            dict_idx = list_dict_idx[ins_idx]
            g, g_metadata, list_srl_edges_metadata, list_ent2ent_metadata, span_idx = self.create_graph(hotpot_instance, list_entities, dict_idx, list_context[ins_idx]['input_ids'], ins_idx)
            self.build_tests(ins_idx, g, span_idx, hotpot_instance['answer'])
            list_graphs.append(g)
            list_span_idx.append(span_idx)
            list_g_metadata.append(g_metadata)
            list_list_srl_edges_metadata.append(list_srl_edges_metadata)
            list_list_ent2ent_metadata.append(list_ent2ent_metadata)
        return list_graphs, list_g_metadata, list_context, list_list_srl_edges_metadata, list_list_ent2ent_metadata, list_span_idx

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
        encoding = self.tokenizer.batch_encode_plus(
                                        list_sent,
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
        # Edges #
        ## Hierarchical
        list_doc2doc = []
        list_doc2sent = []
        list_sent2doc = []
        list_sent2doc = []
        list_sent2srl = []
        list_srl2sent = []
        list_srl2ent = []
        list_ent2srl = []
        list_srl2query = []
        list_query2srl = []
        ### to tokens
        list_doc2tok = []
        list_tok2doc = []
        list_sent2tok = []
        list_tok2sent = []
        list_srl2tok = []
        list_tok2srl = []
        list_ent2tok = []
        list_tok2ent = []
        ## same level
        list_sent2sent = []
        list_srl2srl = []
        list_srl2self = []
        list_srl_tmp2srl = []
        list_srl_loc2srl = []
        list_ent2ent_self = []
        list_token2token = []
        ## multi-hop
        list_doc_multihop = []
        list_sent_multihop = []
        list_srl_multihop = []
        list_ent_multihop = []
        ## srl_edge_metadata (for embeddings)
        list_srl_rel = []
        
        # metadata of each node #
        ## metadata of docs
        list_doc_context_idx = []
        list_doc_st_end_idx = []
        list_doc_lbl = []
        ## metadata of sents
        list_sent_context_idx = []
        list_sent_st_end_idx = []
        list_sent_lbl = []
        ## metadata of srl
        list_srl_context_idx = []
        list_srl_st_end_idx = []
        list_srl_lbl = []
        ## metadata of srl tmp
        list_srl_tmp_context_idx = []
        list_srl_tmp_st_end_idx = []
        list_srl_tmp_lbl = []
        ## metadata of srl loc
        list_srl_loc_context_idx = []
        list_srl_loc_st_end_idx = []
        list_srl_loc_lbl = []
        ## metadata of ents
        list_ent_context_idx = []
        list_ent_st_end_idx = []
        list_ent_lbl = []
        # metadata of tokens
        list_token_context_idx = []
        list_token_st_end_idx = []
        list_token_lbl = []
        # metadata of query
        list_query_st_end_idx = []
        
        list_ent_node2str = []
        dict_ent_str2ent_node = dict()
        token_node_idx = 0
        for i in range(self.max_len):
            list_token_context_idx.append(ins_idx)
            list_token_st_end_idx.append((i, i+1))
            list_token2token.append((i, i))  # lbl: [TOK2TOK_SELF]
            list_token_lbl.append(0)
            token_node_idx += 1
        
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
        doc_node_idx = 0
        sent_node_idx = 0
        srl_node_idx = 0
        srl_loc_node_idx = 0
        srl_tmp_node_idx = 0
        ent_node_idx = 0
        for idx, doc_metadata in enumerate(dict_idx['list_docs_encoding_metadata']):
            # doc node
            current_doc_node = doc_node_idx
            doc_node_idx += 1
            list_doc_nodes.append(current_doc_node)
            # metadata doc
            list_doc_context_idx.append(ins_idx)
            list_doc_st_end_idx.append(doc_metadata['doc_token_st_end_idx'])
            list_doc_lbl.append(1)
            # end metada doc
            # add edges to its tokens
            (st, end) = doc_metadata['doc_token_st_end_idx']
            end = min(end, self.max_len)
            for tok in range(st, end):
                list_doc2tok.append((current_doc_node, tok))  # lbl: [DOC2TOK]
                list_tok2doc.append((tok, current_doc_node))  # lbl: [TOK2DOC]
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
                sent_end = min(sent_end, self.max_len)
                # sent node
                current_sent_node = sent_node_idx
                sent_node_idx += 1
                list_sent_nodes.append(current_sent_node)
                # metadata sent
                list_sent_context_idx.append(ins_idx)
                list_sent_st_end_idx.append((sent_st, sent_end))
                sent_lbl = sent_idx in dict_supporting_docs[doc_title]
                list_sent_lbl.append(sent_lbl)
                # end metada sent
                
                for tok in range(sent_st, sent_end):
                    list_sent2tok.append((current_sent_node, tok))  # lbl: [SENT2TOK]
                    list_tok2sent.append((tok, current_sent_node))  # lbl: [TOK2SENT]
                
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
                    if 'V' not in triple_dict.keys():
                        # we need relations
                        continue
                    if ('ARG0' not in triple_dict.keys()) and ('ARG1' not in triple_dict.keys()) and ('ARG2' not in triple_dict.keys()):
                        # not well-formed triple
                        continue
                    list_arg_nodes = [] # list of all arguments (node idx) in the triple
                    srl_rel = dict()
                    subject_srl = -1
                    current_srl_tmp_node = None
                    current_srl_loc_node = None
                    
                    if 'V' in triple_dict:
                        verb = triple_dict['V']
                        rel_encoded = self.tokenizer.encode(verb, add_special_tokens=False)
                        # find location of the entity in the context (inputs ids)
                        # +sent_st 'cuz I need the index in the full context
                        # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                        try:
                            st_tok_idx = find_sublist_idx(context[sent_st:sent_end], rel_encoded) + sent_st
                            end_tok_idx = min(st_tok_idx + len(rel_encoded), self.max_len)
                            srl_rel['st_tok_idx'] = st_tok_idx
                            srl_rel['end_tok_idx'] = end_tok_idx
                        except:
                            # if the verb cannot be find in the sentence because the sentence was trimmed,
                            # skip this triple
                            continue
                    for arg_type, arg_str in triple_dict.items():
                        try:
                            if arg_type == 'V':
                                # already processed it before the loop
                                pass
                            elif arg_type == 'TMP':
                                arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                                # find location of the entity in the context (inputs ids)
                                # +sent_st 'cuz I need the index in the full context
                                # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                                st_tok_idx = find_sublist_idx(context[sent_st:sent_end], arg_encoded) + sent_st
                                end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                                # metadata
                                list_srl_tmp_context_idx.append(ins_idx)
                                list_srl_tmp_st_end_idx.append((st_tok_idx, end_tok_idx))
                                # node
                                current_srl_tmp_node = srl_tmp_node_idx
                                srl_tmp_node_idx += 1
                            elif arg_type == 'LOC':
                                arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                                # find location of the entity in the context (inputs ids)
                                # +sent_st 'cuz I need the index in the full context
                                # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                                st_tok_idx = find_sublist_idx(context[sent_st:sent_end], arg_encoded) + sent_st
                                end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                                # metadata
                                list_srl_loc_context_idx.append(ins_idx)
                                list_srl_loc_st_end_idx.append((st_tok_idx, end_tok_idx))
                                # node
                                current_srl_loc_node = srl_loc_node_idx
                                srl_loc_node_idx += 1
                            else:
                                if 'ARG' in arg_type and subject_srl == -1:
                                    subject_srl = srl_node_idx
                                arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                                arg_str = self.tokenizer.decode(arg_encoded)
                                # find location of the entity in the context (inputs ids)
                                # +sent_st 'cuz I need the index in the full context
                                # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                                st_tok_idx = find_sublist_idx(context[sent_st:sent_end], arg_encoded) + sent_st
                                end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                                # metadata ent
                                list_srl_context_idx.append(ins_idx)
                                list_srl_st_end_idx.append((st_tok_idx, end_tok_idx))
                                #list_node_type.append(node_type2idx['srl'])
                                srl_lbl = self.srl_with_ans(arg_str, ans_detokenized, sent_lbl)
                                list_srl_lbl.append(srl_lbl)
                                # ent node
                                current_srl_node = srl_node_idx
                                srl_node_idx += 1
                                list_arg_nodes.append(current_srl_node)
#                                 print("SRL:", current_srl_node, arg_str)
                                # srl <-> token
                                for tok in range(st_tok_idx, end_tok_idx):
                                    list_srl2tok.append((current_srl_node, tok))  # lbl: [SRL2TOK]
                                    list_tok2srl.append((tok, current_srl_node))  # lbl: [TOK2SRL]
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
                                        
                                        lbl = (fuzz.token_set_ratio(ans_detokenized, ent) >= 90)
                                        if (not srl_lbl) and lbl:
                                            srl_lbl = True
                                            list_lbl[-1] = srl_lbl
                                        # ent node
                                        if ent in dict_ent_str2ent_node:
                                            current_ent_node = dict_ent_str2ent_node[ent]
                                        else:
                                            current_ent_node = ent_node_idx
                                            ent_node_idx += 1
                                            dict_ent_str2ent_node[ent] = current_ent_node
                                            list_ent_node2str.append(ent)
                                            # metadata ent
                                            list_ent_context_idx.append(ins_idx)
                                            list_ent_st_end_idx.append((st_tok_idx, end_tok_idx))
                                            list_ent_lbl.append(lbl)
                                        
#                                         print("ent:", current_ent_node, ent)
#                                         print("ent2srl", current_ent_node, current_srl_node)
                                        # add edges (srl_arg -> ent and ent -> srl_arg)
                                        list_srl2ent.append((current_srl_node, current_ent_node))  # lbl: [SRL2ENT]
                                        list_ent2srl.append((current_ent_node, current_srl_node))  # lbl: [ENT2SRL]
                                        # self edge ent
                                        list_ent2ent_self.append((current_ent_node, current_ent_node))  # lbl: [ENT2ENT_SELF]
                                        
                                        # srl -> token
                                        for tok in range(st_tok_idx, end_tok_idx):
                                            list_ent2tok.append((current_ent_node, tok))  # lbl: [ENT2TOK]
                                            list_tok2ent.append((tok, current_ent_node))   # lbl: [TOK2ENT]
                #                         print("#### ent ", lbl, self.tokenizer.decode(context[st_tok_idx:end_tok_idx]))
                                    except:
                #                         print(ent)
                #                         print("### sent", sent_str)
                                        pass
                            
                                dict_srl_metadata[current_srl_node] = list_srl_ent
                        except:
                            pass
                    # SRL <-> SRL (bidirectional) # lbl: [SRL2SRL]
                    for arg1_idx, arg1 in enumerate(list_arg_nodes):
                        for arg2 in list_arg_nodes:
                            if arg1 != arg2:
                                rel_type = 1
                                if arg2 == subject_srl:
                                    rel_type = -1
                                list_srl_rel.append({'rel_type': rel_type, 
                                                     'span_idx': [srl_rel['st_tok_idx'],               
                                                                  srl_rel['end_tok_idx']] 
                                                    })
                                list_srl2srl.append((arg1, arg2))
#                     list_srl2srl.extend([(arg1, arg2) for arg1 in list_arg_nodes
#                                                       for arg2 in list_arg_nodes])
                    # SRL_TMP -> SRL # lbl: [SRL_TMP2SRL]
                    if current_srl_tmp_node is not None:
                        list_srl_tmp2srl.extend([(current_srl_tmp_node, arg1) for arg1 in list_arg_nodes])
                    # SRL_LOC -> SRL # lbl: [SRL_LOC2SRL]
                    if current_srl_loc_node is not None:
                        list_srl_loc2srl.extend([(current_srl_loc_node, arg1) for arg1 in list_arg_nodes])
                    # srl_arg -> sent  # lbl: [SRL2SENT]
                    list_srl2sent.extend([(arg, current_sent_node) for arg in list_arg_nodes])        
                    # sent -> arg_srl  # lbl: [SENT2SRL]
                    list_sent2srl.extend([(current_sent_node, arg) for arg in list_arg_nodes])
                # sent -> doc  # lbl: [SENT2DOC]
                list_sent2doc.append((current_sent_node, current_doc_node))
                # doc -> sent  # lbl: [DOC2SENT]
                list_doc2sent.append((current_doc_node, current_sent_node))
            # sequentially connected sent nodes # lbl: [SENT2SENT] 
            list_sent2sent.extend([(u, v) for u in list_sent_nodes for v in list_sent_nodes if u <= v])
        # self-edges to the docs
        list_doc2doc.extend([(u, u) for u in list_doc_nodes])  # lbl: [DOC2DOC_SELF]
#         # fully connected doc nodes    
#         list_doc2doc.extend([(u, v) for u in list_doc_nodes for v in list_doc_nodes if u != v])
#         list_doc2doc.extend([(v, u) for u in list_doc_nodes for v in list_doc_nodes if u != v])
        
        ############ Query node ################
        for key, triple_dict in self.dict_ins_query_triples[str(ins_idx)].items():
            if not self.valid_srl_triple(triple_dict):
                continue

            list_arg_nodes = [] # list of all arguments (node idx) in the triple
            srl_rel = dict()
            subject_srl = -1
            current_srl_tmp_node = None
            current_srl_loc_node = None

            if 'V' in triple_dict:
                verb = triple_dict['V']
                rel_encoded = self.tokenizer.encode(verb, add_special_tokens=False)
                # find location of the entity in the context (inputs ids)
                # +sent_st 'cuz I need the index in the full context
                # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                try:
                    st_tok_idx = find_sublist_idx(context[q_st:q_end], rel_encoded) + q_st
                    end_tok_idx = min(st_tok_idx + len(rel_encoded), self.max_len)
                    srl_rel['st_tok_idx'] = st_tok_idx
                    srl_rel['end_tok_idx'] = end_tok_idx
                except:
                    # if the verb cannot be find in the sentence because the sentence was trimmed,
                    # skip this triple
                    continue

            for arg_type, arg_str in triple_dict.items():
                try:
                    if arg_type == 'V':
                        # already processed it before the loop
                        continue
                    elif arg_type == 'TMP':
                        arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                        # find location of the entity in the context (inputs ids)
                        # +sent_st 'cuz I need the index in the full context
                        # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                        st_tok_idx = find_sublist_idx(context[q_st:q_end], arg_encoded) + q_st
                        end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                        # metadata
                        list_srl_tmp_context_idx.append(ins_idx)
                        list_srl_tmp_st_end_idx.append((st_tok_idx, end_tok_idx))
                        # node
                        current_srl_tmp_node = srl_tmp_node_idx
                        srl_tmp_node_idx += 1
                    elif arg_type == 'LOC':
                        arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                        # find location of the entity in the context (inputs ids)
                        # +sent_st 'cuz I need the index in the full context
                        # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                        st_tok_idx = find_sublist_idx(context[sent_st:sent_end], arg_encoded) + sent_st
                        end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                        # metadata
                        list_srl_loc_context_idx.append(ins_idx)
                        list_srl_loc_st_end_idx.append((st_tok_idx, end_tok_idx))
                        # node
                        current_srl_loc_node = srl_loc_node_idx
                        srl_loc_node_idx += 1
                    else:
                        if 'ARG' in arg_type and subject_srl == -1:
                            subject_srl = srl_node_idx
                        arg_encoded = self.tokenizer.encode(arg_str, add_special_tokens=False)
                        arg_str = self.tokenizer.decode(arg_encoded)
                        # find location of the entity in the context (inputs ids)
                        # +sent_st 'cuz I need the index in the full context
                        # if I search using the full context instead of its sentence I may get a wrong entity (the entity may appear many times, including in the question)
                        st_tok_idx = find_sublist_idx(context[q_st:q_end], arg_encoded) + q_st
                        end_tok_idx = min(st_tok_idx + len(arg_encoded), self.max_len)
                        # metadata ent
                        list_srl_context_idx.append(ins_idx)
                        list_srl_st_end_idx.append((st_tok_idx, end_tok_idx))
                        #list_node_type.append(node_type2idx['srl'])
                        srl_lbl = self.srl_with_ans(arg_str, ans_detokenized, sent_lbl)
                        list_srl_lbl.append(srl_lbl)
                        # ent node
                        current_srl_node = srl_node_idx
                        srl_node_idx += 1
                        list_arg_nodes.append(current_srl_node)

                        # srl <-> token
                        for tok in range(st_tok_idx, end_tok_idx):
                            list_srl2tok.append((current_srl_node, tok))  # lbl: [SRL2TOK]
                            list_tok2srl.append((tok, current_srl_node))  # lbl: [TOK2SRL]
                        #######################################################
                        # for each entity
                        list_srl_ent = []
                        for ent in self.list_entities_query[ins_idx]:
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
                                st_tok_idx = find_sublist_idx(context[q_st:q_end], ent_encoded) + q_st
                                end_tok_idx = min(st_tok_idx + len(ent_encoded), self.max_len)

                                lbl = (fuzz.token_set_ratio(ans_detokenized, ent) >= 90)
                                if (not srl_lbl) and lbl:
                                    srl_lbl = True
                                    list_lbl[-1] = srl_lbl
                                # ent node
                                if ent in dict_ent_str2ent_node:
                                    current_ent_node = dict_ent_str2ent_node[ent]
                                else:
                                    current_ent_node = ent_node_idx
                                    ent_node_idx += 1
                                    dict_ent_str2ent_node[ent] = current_ent_node
                                    list_ent_node2str.append(ent)
                                    # metadata ent
                                    list_ent_context_idx.append(ins_idx)
                                    list_ent_st_end_idx.append((st_tok_idx, end_tok_idx))
                                    list_ent_lbl.append(lbl)
                                    
                                # add edges (srl_arg -> ent and ent -> srl_arg)
                                list_srl2ent.append((current_srl_node, current_ent_node))  # lbl: [SRL2ENT]
                                list_ent2srl.append((current_ent_node, current_srl_node))  # lbl: [ENT2SRL]
                                # self edge ent
                                list_ent2ent_self.append((current_ent_node, current_ent_node))  # lbl: [ENT2ENT_SELF]
                                
                                # srl -> token
                                for tok in range(st_tok_idx, end_tok_idx):
                                    list_ent2tok.append((current_ent_node, tok))  # lbl: [ENT2TOK]
                                    list_tok2ent.append((tok, current_ent_node))   # lbl: [TOK2ENT]
        #                         print("#### ent ", lbl, self.tokenizer.decode(context[st_tok_idx:end_tok_idx]))
                            except:
        #                         print(ent)
        #                         print("### sent", sent_str)
                                pass

                        dict_srl_metadata[current_srl_node] = list_srl_ent
                except:
                    pass
            # SRL <-> SRL (bidirectional) # lbl: [SRL2SRL]
            for arg1_idx, arg1 in enumerate(list_arg_nodes):
                for arg2 in list_arg_nodes:
                    if arg1 != arg2:
                        rel_type = 1
                        if arg2 == subject_srl:
                            rel_type = -1
                        list_srl_rel.append({'rel_type': rel_type, 
                                            'span_idx': [srl_rel['st_tok_idx'],               
                                                        srl_rel['end_tok_idx']] 
                                            })
                        list_srl2srl.append((arg1, arg2))
            # SRL_TMP -> SRL # lbl: [SRL_TMP2SRL]
            if current_srl_tmp_node is not None:
                list_srl_tmp2srl.extend([(current_srl_tmp_node, arg1) for arg1 in list_arg_nodes])
            # SRL_LOC -> SRL # lbl: [SRL_LOC2SRL]
            if current_srl_loc_node is not None:
                list_srl_loc2srl.extend([(current_srl_loc_node, arg1) for arg1 in list_arg_nodes])
            # srl_arg -> query  # lbl: [SRL2QUERY]
            list_srl2query.extend([(arg, 0) for arg in list_arg_nodes])        
            # query -> arg_srl  # lbl: [QUERY2SRL]
            list_query2srl.extend([(0, arg) for arg in list_arg_nodes])
            (q_st, q_end) = dict_idx['q_token_st_end_idx']
            # metadata sent
            list_query_st_end_idx.append((q_st, q_end))
        ############ END Query node ################
        list_ent_nodes = list(range(ent_node_idx))
        (list_ent_multihop, 
         list_srl_multihop,
         list_sent_multihop,
         list_sent2query_multihop,
         list_query2sent_multihop) = self.multi_hop_edges(list_ent_nodes, 
                                                          list_ent_node2str, 
                                                          list_ent2srl,                
                                                          list_srl2sent,
                                                          list_srl2query,
                                                          90)
        # make the heterogenous graph
        list_srl2self = [(v, v) for v in range(srl_node_idx)]
        # create ent rel using SRL predicates
        list_ent2ent_rel, list_ent2ent_metadata = self.compute_ent_relations(list_srl2srl, 
                                                                             list_srl2ent, 
                                                                             list_srl_rel)
        dict_edges = {
#             ('doc', 'doc2sent', 'sent'): list_doc2sent,  # lbl: [DOC2SENT]
                     ('sent', 'sent2doc', 'doc'): list_sent2doc,  # lbl: [SENT2DOC]
#                      ('sent', 'sent2srl', 'srl'): list_sent2srl,  # lbl: [SENT2SRL]
                     ('srl', 'srl2sent', 'sent'): list_srl2sent,  # lbl: [SRL2SENT]
#                      ('srl', 'srl2ent', 'ent'): list_srl2ent,     # lbl: [SRL2ENT]
                      ('ent', 'ent2srl', 'srl'): list_ent2srl,     # lbl: [ENT2SRL]
                      
                     # to token
#                      ('doc', 'doc2tok', 'tok'): list_doc2tok,     # lbl: [DOC2TOK]
#                      ('tok', 'tok2doc', 'doc'): list_tok2doc,     # lbl: [TOK2DOC]
#                      ('sent', 'sent2tok', 'tok'): list_sent2tok,  # lbl: [SENT2TOK]
#                      ('tok', 'tok2sent', 'sent'): list_tok2sent,  # lbl: [TOK2SENT]
                     ('srl', 'srl2tok', 'tok'): list_srl2tok,     # lbl: [SRL2TOK]
#                      ('tok', 'tok2srl', 'srl'): list_tok2srl,     # lbl: [TOK2SRL]
                     ('ent', 'ent2tok', 'tok'): list_ent2tok,     # lbl: [ENT2TOK]
#                      ('tok', 'tok2ent', 'ent'): list_tok2ent,     # lbl: [TOK2ENT]
                     # end hierarchical
                     # same-level edges
                     ('doc', 'doc2doc_self', 'doc'): list_doc2doc,         # lbl: [DOC2DOC_SELF]
                     ('sent', 'sent2sent', 'sent'): list_sent2sent,   # lbl: [SENT2SENT]
                     ('srl', 'srl2srl', 'srl'): list_srl2srl,         # lbl: [SRL2SRL]
                     ('srl', 'srl2self', 'srl'): list_srl2self,         # lbl: [SRL2SELF]
                     ('ent', 'ent2ent_self', 'ent'): list_ent2ent_self,         # lbl: [ENT2ENT_SELF]
                     ('ent', 'ent2ent_rel', 'ent'): list_ent2ent_rel,
                     ('tok', 'token2token_self', 'tok'): list_token2token, # lbl: [TOK2TOK_SELF]
                     # multi-hop edges
                     ('ent', 'ent_multihop', 'ent'): list_ent_multihop,
                     ('srl', 'srl_multihop', 'srl'): list_srl_multihop,
                     ('sent', 'sent_multihop', 'sent'): list_sent_multihop,
                    }
        if list_srl_loc2srl != []:
            dict_edges[('srl_loc', 'srl_loc2srl', 'srl')] = list_srl_loc2srl  # lbl: [SRL_LOC2SRL]
        if list_srl_tmp2srl != []:
            dict_edges[('srl_tmp', 'srl_tmp2srl', 'srl')] = list_srl_tmp2srl  # lbl: [SRL_TMP2SRL]
        if list_sent2query_multihop != []:
            dict_edges[('sent', 'sent2query_multihop', 'query')] = list_sent2query_multihop
        if list_query2sent_multihop != []:
            dict_edges[('query', 'query2sent_multihop', 'sent')] = list_query2sent_multihop
        if list_srl2query != []:
            dict_edges[('srl', 'srl2query', 'query')] = list_srl2query
#         if list_query2srl != []:
#             dict_edges[('query', 'query2srl', 'srl')] = list_query2srl
        graph = dgl.heterograph(dict_edges)
        graph_metadata = dict()
        # doc metadata
        graph_metadata['doc'] = dict()
        graph_metadata['doc']['st_end_idx'] =  np.array(list_doc_st_end_idx)
        graph_metadata['doc']['list_context_idx'] = np.array(list_doc_context_idx).reshape(-1,1)
        graph_metadata['doc']['labels'] = np.array(list_doc_lbl).reshape(-1,1)
        # sent metadata
        graph_metadata['sent'] = dict()
        graph_metadata['sent']['st_end_idx'] =  np.array(list_sent_st_end_idx)
        graph_metadata['sent']['list_context_idx'] = np.array(list_sent_context_idx).reshape(-1,1)
        graph_metadata['sent']['labels'] = np.array(list_sent_lbl).reshape(-1,1)
        # srl metadata
        graph_metadata['srl'] = dict()
        graph_metadata['srl']['st_end_idx'] =  np.array(list_srl_st_end_idx)
        graph_metadata['srl']['list_context_idx'] = np.array(list_srl_context_idx).reshape(-1,1)
        graph_metadata['srl']['labels'] = np.array(list_srl_lbl).reshape(-1,1)
        # srl_loc metadata
        if list_srl_loc2srl != []:
            graph_metadata['srl_loc'] = dict()
            graph_metadata['srl_loc']['st_end_idx'] =  np.array(list_srl_loc_st_end_idx)
            graph_metadata['srl_loc']['list_context_idx'] = np.array(list_srl_loc_context_idx).reshape(-1,1)
        # srl_tmp metadata
        if list_srl_tmp2srl != []:
            graph_metadata['srl_tmp'] = dict()
            graph_metadata['srl_tmp']['st_end_idx'] =  np.array(list_srl_tmp_st_end_idx)
            graph_metadata['srl_tmp']['list_context_idx'] = np.array(list_srl_tmp_context_idx).reshape(-1,1) 
        # ent metadata
        graph_metadata['ent'] = dict()
        graph_metadata['ent']['st_end_idx'] =  np.array(list_ent_st_end_idx)
        graph_metadata['ent']['list_context_idx'] = np.array(list_ent_context_idx).reshape(-1,1)
        graph_metadata['ent']['labels'] = np.array(list_ent_lbl).reshape(-1,1)
        # token metadata
        graph_metadata['tok'] = dict()
        graph_metadata['tok']['st_end_idx'] =  np.array(list_token_st_end_idx)
        graph_metadata['tok']['list_context_idx'] = np.array(list_token_context_idx).reshape(-1,1)
        graph_metadata['tok']['labels'] = np.array(list_token_lbl).reshape(-1,1)
        # query metadata
        if list_query_st_end_idx != []:
            graph_metadata['query'] = dict()
            graph_metadata['query']['st_end_idx'] =  np.array(list_query_st_end_idx)
#         # doc metadata
#         graph.nodes['doc'].data['st_end_idx'] =  np.array(list_doc_st_end_idx)
#         graph.nodes['doc'].data['list_context_idx'] = np.array(list_doc_context_idx).reshape(-1,1)
#         graph.nodes['doc'].data['labels'] = np.array(list_doc_lbl).reshape(-1,1)
#         # sent metadata
#         graph.nodes['sent'].data['st_end_idx'] =  np.array(list_sent_st_end_idx)
#         graph.nodes['sent'].data['list_context_idx'] = np.array(list_sent_context_idx).reshape(-1,1)
#         graph.nodes['sent'].data['labels'] = np.array(list_sent_lbl).reshape(-1,1)
#         # srl metadata
#         graph.nodes['srl'].data['st_end_idx'] =  np.array(list_srl_st_end_idx)
#         graph.nodes['srl'].data['list_context_idx'] = np.array(list_srl_context_idx).reshape(-1,1)
#         graph.nodes['srl'].data['labels'] = np.array(list_srl_lbl).reshape(-1,1)
#         # ent metadata
#         graph.nodes['ent'].data['st_end_idx'] =  np.array(list_ent_st_end_idx)
#         graph.nodes['ent'].data['list_context_idx'] = np.array(list_ent_context_idx).reshape(-1,1)
#         graph.nodes['ent'].data['labels'] = np.array(list_ent_lbl).reshape(-1,1)
#         # token metadata
#         graph.nodes['tok'].data['st_end_idx'] =  np.array(list_token_st_end_idx)
#         graph.nodes['tok'].data['list_context_idx'] = np.array(list_token_context_idx).reshape(-1,1)
#         graph.nodes['tok'].data['labels'] = np.array(list_token_lbl).reshape(-1,1)
        
        return graph, graph_metadata, list_srl_rel, list_ent2ent_metadata, (ans_st_idx, ans_end_idx)

    def compute_ent_relations(self, list_srl2srl, list_srl2ent, list_srl_rel_metadata):
        # aux data structure
        dict_srl2ent = dict()
        for (srl, e) in list_srl2ent:
            if srl in dict_srl2ent:
                dict_srl2ent[srl].append(e)
            else:
                dict_srl2ent[srl] = [e]
        # algorithm starts here
        # for each srl rel, look for their children and inherit that relation
        list_ent_rel = []
        list_ent_rel_metadata = []
        for i, (srl1, srl2) in enumerate(list_srl2srl):
            if srl1 not in dict_srl2ent:
                continue
            if srl2 not in dict_srl2ent:
                continue
            list_e1 = dict_srl2ent[srl1]
            list_e2 = dict_srl2ent[srl2]
            for e1 in list_e1:
                for e2 in list_e2:
                    list_ent_rel.append((e1,e2))
                    list_ent_rel_metadata.append(list_srl_rel_metadata[i])
        return list_ent_rel, list_ent_rel_metadata
        
    def srl_with_ans(self, srl_arg: str, ans:str, in_supp_sent: bool) -> bool:
        return (in_supp_sent and ans != "yes" and ans != "no" and
                ((ans in srl_arg) or (ans[:-1] in srl_arg) or (fuzz.token_set_ratio(srl_arg, ans) >= 90)))
        
    def valid_srl_triple(self, triple_dict):
        if 'V' not in triple_dict.keys():
            # we need relations
            return False
        if ('ARG0' not in triple_dict.keys()) and ('ARG1' not in triple_dict.keys()) and ('ARG2' not in triple_dict.keys()):
            # not well-formed triple
            return False
        return True
  
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
    
    def create_common_entity_edges_sent_lvl(self, dict_sent_node2metadata: dict, list_hotpot_instance_ner: list) -> list:
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
    
    def multi_hop_edges(self, list_ent_nodes, list_ent_str, list_ent2srl, list_srl2sent, list_srl2query, threshold):
#         print("######################## num ent", max(list_ent_nodes))
#         print("ent2srl", list_ent2srl)
#         print("srl2sent", list_srl2sent)
#         print(list_ent_str)
        dict_ent2srl = dict()
        dict_srl2sent = dict()
        dict_srl2query = dict()
        for (e, srl) in list_ent2srl:
            dict_ent2srl[e] = srl
        for (srl, sent) in list_srl2sent:
            dict_srl2sent[srl] = sent
        for (srl, query) in list_srl2query:
            dict_srl2query[srl] = query

        list_ent_multihop = []
        list_srl_multihop = []
        list_sent_multihop = []
        list_sent2query_multihop = []
        list_query2sent_multihop = []
        for e1 in list_ent_nodes:
            for e2 in list_ent_nodes:
                if e1 < e2:
                    e1_str = list_ent_str[e1]
                    e2_str = list_ent_str[e2]
                    if fuzz.token_set_ratio(e1_str, e2_str) >= threshold:
#                         print(e1_str, e2_str, fuzz.token_set_ratio(e1_str, e2_str))
                        # ent
                        list_ent_multihop.append((e1, e2))  # LBL: [ENT2ENT_MULTIHOP]
                        list_ent_multihop.append((e2, e1))  # LBL: [ENT2ENT_MULTIHOP]
                        # srl
                        srl1 = dict_ent2srl[e1]
                        srl2 = dict_ent2srl[e2]
                        list_srl_multihop.append((srl1, srl2))  # LBL: [SRL2SRL_MULTIHOP]
                        list_srl_multihop.append((srl2, srl1))  # LBL: [SRL2SRL_MULTIHOP]
                        # sent
                        sent1 = None
                        sent2 = None
                        q_node = None
                        if srl1 in dict_srl2sent:
                            sent1 = dict_srl2sent[srl1]
                        elif srl1 in dict_srl2query:
                            q_node = dict_srl2query[srl]
                        if srl2 in dict_srl2sent:
                            sent2 = dict_srl2sent[srl2]
                        elif srl2 in dict_srl2query:
                            q_node = dict_srl2query[srl]
                        if sent1 is not None and sent2 is not None:
                            list_sent_multihop.append((sent1, sent2))  # LBL: [SENT2SENT_MULTIHOP]
                            list_sent_multihop.append((sent2, sent1))  # LBL: [SENT2SENT_MULTIHOP]
#                             print(sent1, sent2, srl1, srl2, e1, e2, e1_str, e2_str, fuzz.token_set_ratio(e1_str, e2_str))
                        elif sent1 is None and (sent2 is not None and q_node is not None):
                            list_sent2query_multihop.append((sent2, query))
                            list_query2sent_multihop.append((query, sent2))
                        elif sent2 is None and (sent1 is not None and q_node is not None):
                            list_sent2query_multihop.append((sent1, query))
                            list_query2sent_multihop.append((query, sent1))
        
        list_ent_multihop = list(set(list_ent_multihop))
#         for (e1, e2) in list_ent_multihop:
#             print(list_ent_str[e1], list_ent_str[e2], e1, e2)
#         print(list_ent_multihop)
        list_srl_multihop = list(set(list_srl_multihop))
        list_sent_multihop = list(set(list_sent_multihop))
        list_sent2query_multihop = list(set(list_sent2query_multihop))
        list_query2sent_multihop = list(set(list_query2sent_multihop))
        return list_ent_multihop, list_srl_multihop, list_sent_multihop, list_sent2query_multihop, list_query2sent_multihop


# -


def add_metadata2graph(graph, metadata):
    for (node, dict_node) in metadata.items():
        for (k, v) in dict_node.items():
            graph.nodes[node].data[k] = torch.tensor(v)
    return graph


train_dataset = Dataset(hotpot_train[0:40000], list_hotpot_train_ner, dict_ins_doc_sent_srl_triples,
                        dict_ins_query_srl_triples_training, list_ent_query_training, batch_size=1)
(list_graphs, 
 list_g_metadata,
 list_context,
 list_list_srl_edges_metadata,
 list_list_ent2ent_metadata,
 list_span_idx) = train_dataset.create_dataloader()


def print_ent_rel():
    (list_u, list_v, _) = list_graphs[0].all_edges(form='all', etype='ent2ent_rel')
    for i, u in enumerate(list_u):
        st = list_g_metadata[0]['ent']['st_end_idx'][u][0]
        end = list_g_metadata[0]['ent']['st_end_idx'][u][1]
        st_v = list_g_metadata[0]['ent']['st_end_idx'][list_v[i]][0]
        end_v = list_g_metadata[0]['ent']['st_end_idx'][list_v[i]][1]
        # rel
        st_r, end_r = list_list_ent2ent_metadata[0][i]['span_idx']
        r_type = list_list_ent2ent_metadata[0][i]['rel_type']
        print(train_dataset.tokenizer.decode(list_context[0]['input_ids'][st:end]), ", ", 
             r_type, train_dataset.tokenizer.decode(list_context[0]['input_ids'][st_r:end_r]), ", ",
             train_dataset.tokenizer.decode(list_context[0]['input_ids'][st_v:end_v]))


# +
# edges = 0
# for g in list_graphs:
#     edges += g.number_of_edges('ent2ent_multihop')
# print(edges/len(list_graphs)/2) # divided by two because edges are bidirectional
# -

for g_idx, list_dict_edge in enumerate(list_list_srl_edges_metadata):
    list_graphs[g_idx].edges['srl2srl'].data['rel_type'] = torch.tensor([edge['rel_type'] for edge in list_dict_edge])
    list_graphs[g_idx].edges['srl2srl'].data['span_idx'] = torch.tensor([edge['span_idx'] for edge in list_dict_edge])

for g_idx, list_dict_edge in enumerate(list_list_ent2ent_metadata):
    list_graphs[g_idx].edges['ent2ent_rel'].data['rel_type'] = torch.tensor([edge['rel_type'] for edge in list_dict_edge])
    list_graphs[g_idx].edges['ent2ent_rel'].data['span_idx'] = torch.tensor([edge['span_idx'] for edge in list_dict_edge])

training_path = os.path.join(data_path, 'processed/training/heterog_20200830_bottomup')
training_graph_path = os.path.join(training_path, 'graphs')
training_metadata_path = os.path.join(training_path, 'metadata')

for i, g in enumerate(tqdm(list_graphs)):
    with open(os.path.join(training_graph_path, "graph" + str(i) + ".bin"), "wb" ) as f:
        pickle.dump(g, f)
    with open( os.path.join(training_metadata_path, "metadata" + str(i) + ".bin"), "wb" ) as f:
        pickle.dump(list_g_metadata[i], f)
    # separate the metadata from the graph to store it (do not add metadata in the first place)
    # weird problem if the metadata is inside the graph


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# +
# list_graph_metadata = list(zip(list_graphs, list_g_metadata))
# for i, (graph, metadata) in enumerate(tqdm(list_graph_metadata)):
#     graph = add_metadata2graph(graph, metadata)
#     f = os.path.join(training_graph_path, "graph" + str(i) + ".bin")
#     dgl.save_graphs(f, [graph])

# +
list_input_ids = [context['input_ids'] for context in list_context]
list_token_type_ids = [context['token_type_ids'] for context in list_context]
list_attention_masks = [context['attention_mask'] for context in list_context]

tensor_input_ids = torch.tensor(list_input_ids)
tensor_token_type_ids = torch.tensor(list_token_type_ids)
tensor_attention_masks = torch.tensor(list_attention_masks)

torch.save(tensor_input_ids, os.path.join(training_path, 'tensor_input_ids.p'))
torch.save(tensor_token_type_ids, os.path.join(training_path, 'tensor_token_type_ids.p'))
torch.save(tensor_attention_masks, os.path.join(training_path, 'tensor_attention_masks.p'))
# -

with open(os.path.join(training_path, 'list_span_idx.p'), 'wb') as f:
    pickle.dump(list_span_idx, f)



# dev data
with open(os.path.join(data_path, hotpotqa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
    hotpot_dev = json.load(f)
with open(os.path.join(data_path, intermediate_dev_data_path, "list_hotpot_ner_no_coref_dev.p"), "rb") as f:
    list_hotpot_dev_ner = pickle.load(f)
with open(os.path.join(data_path, intermediate_dev_data_path, "dict_ins_doc_sent_srl_triples_dev.json"), 'r') as f:
    dict_ins_doc_sent_srl_triples_dev = json.load(f)
with open(os.path.join(data_path, intermediate_dev_data_path, "dict_ins_query_srl_triples.json"), "r") as f:
    dict_ins_query_srl_triples_dev = json.load(f)
with open(os.path.join(data_path, intermediate_train_data_path, "list_ent_query_dev.p"), "rb") as f:
    list_ent_query_dev = pickle.load(f)
print("Dev data loaded")

# +
dev_dataset = Dataset(hotpot_dev, list_hotpot_dev_ner, dict_ins_doc_sent_srl_triples_dev,
                      dict_ins_query_srl_triples_dev, list_ent_query_dev, batch_size=1)
(list_graphs, 
 list_g_metadata,
 list_context,
 list_list_srl_edges_metadata,
 list_list_ent2ent_metadata,
 list_span_idx) = dev_dataset.create_dataloader()

for g_idx, list_dict_edge in enumerate(list_list_srl_edges_metadata):
    list_graphs[g_idx].edges['srl2srl'].data['rel_type'] = torch.tensor([edge['rel_type'] for edge in list_dict_edge])
    list_graphs[g_idx].edges['srl2srl'].data['span_idx'] = torch.tensor([edge['span_idx'] for edge in list_dict_edge])

for g_idx, list_dict_edge in enumerate(list_list_ent2ent_metadata):
    list_graphs[g_idx].edges['ent2ent_rel'].data['rel_type'] = torch.tensor([edge['rel_type'] for edge in list_dict_edge])
    list_graphs[g_idx].edges['ent2ent_rel'].data['span_idx'] = torch.tensor([edge['span_idx'] for edge in list_dict_edge])

list_input_ids = [context['input_ids'] for context in list_context]
list_token_type_ids = [context['token_type_ids'] for context in list_context]
list_attention_masks = [context['attention_mask'] for context in list_context]
tensor_input_ids = torch.tensor(list_input_ids)
tensor_token_type_ids = torch.tensor(list_token_type_ids)
tensor_attention_masks = torch.tensor(list_attention_masks)
# -

dev_path = os.path.join(data_path, 'processed/dev/heterog_20200830_bottomup')
dev_graph_path = os.path.join(dev_path, 'graphs')
dev_metadata_path = os.path.join(dev_path, 'metadata')

for i, g in enumerate(list_graphs):
    with open(os.path.join(dev_graph_path, "graph" + str(i) + ".bin"), "wb" ) as f:
        pickle.dump(g, f)
    with open( os.path.join(dev_metadata_path, "metadata" + str(i) + ".bin"), "wb" ) as f:
        pickle.dump(list_g_metadata[i], f)
    # separate the metadata from the graph to store it (do not add metadata in the first place)

torch.save(tensor_input_ids, os.path.join(dev_path, 'tensor_input_ids.p'))
torch.save(tensor_token_type_ids, os.path.join(dev_path, 'tensor_token_type_ids.p'))
torch.save(tensor_attention_masks, os.path.join(dev_path, 'tensor_attention_masks.p'))
with open(os.path.join(dev_path, 'list_span_idx.p'), 'wb') as f:
    pickle.dump(list_span_idx, f)


