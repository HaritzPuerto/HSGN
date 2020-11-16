#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor


class SRL():
    def __init__(self, device='cuda'):
        self.predictor = None
        if device == 'cuda':
            self.predictor = Predictor.from_path("models/srl/srl_model.tar.gz", cuda_device=0)
        else:
            self.predictor = Predictor.from_path("models/srl/srl_model.tar.gz")

    def sentence2srl_args(self, sentence: str) -> (list, list):
        '''
        Returns a triple:
            - dictionary of the form: argX -> [int] (argument to spacy token index)
            - list of spacy tokens
        '''
        srl_instance = self.predictor.predict_tokenized(sentence.split())
        list_dict_arg2idx = []
        for verb in srl_instance['verbs']:
            dict_tag2idx = dict()
            for i, tag in enumerate(verb['tags']):
                if tag != 'O':
                    tag = tag.split("-")[-1]
                    if tag in dict_tag2idx:
                        dict_tag2idx[tag].append(i)
                    else:
                        dict_tag2idx[tag] = [i]
            list_dict_arg2idx.append(dict_tag2idx)
        return list_dict_arg2idx, srl_instance['words']

    def extract_srl_from_query(self, hotpot):
        dict_ins_query_srl_triples = dict()
        for ins_idx, hotpot_instance in enumerate(tqdm(hotpot)):
            query = hotpot_instance['question']
            list_dict_arg2idx, list_tokens = self.sentence2srl_args(query)
            # TRIPLES
            dict_srl_triples = dict()
            for triple_idx, dict_arg2idx in enumerate(list_dict_arg2idx):
                # ARGUMENTS
                dict_triple = dict()
                if len(dict_arg2idx.keys()) <= 1:
                    # avoid triples where we only have the verb. They are not real srl triples
                    continue
                for key_idx, (arg, list_token_idx) in enumerate(dict_arg2idx.items()):
                    srl_tokenized_span = [list_tokens[idx] for idx in list_token_idx]
                    srl_span = " ".join(srl_tokenized_span)
                    dict_triple[arg] = srl_span
                dict_srl_triples[triple_idx] = dict_triple
            dict_ins_query_srl_triples[ins_idx] = dict_srl_triples
        return dict_ins_query_srl_triples

    def extract_srl(self, hotpot, dict_ins2dict_doc2pred):
        dict_ins_doc_sent_srl_triples = dict()
        for ins_idx, hotpot_instance in enumerate(tqdm(hotpot)):
            dict_doc_sent_srl_triples = dict()
            for doc_idx, (doc_title, doc) in enumerate(hotpot_instance['context']):
                if dict_ins2dict_doc2pred[ins_idx][doc_idx] == 0:
                    continue
                ####### Process sentences #######            
                dict_sent_srl_triples = dict()
                for sent_idx, sent in enumerate(doc):
                    ##### SRL Level ######
                    #get SRL instance
                    list_dict_arg2idx, list_tokens = self.sentence2srl_args(sent)

                    ######################################## TRIPLES
                    dict_srl_triples = dict()
                    for triple_idx, dict_arg2idx in enumerate(list_dict_arg2idx):
                        ###################################### ARGUMENTS
                        dict_triple = dict()
                        if len(dict_arg2idx.keys()) <= 1:
                            # avoid triples where we only have the verb. They are not real srl triples
                            continue
                        for key_idx, (arg, list_token_idx) in enumerate(dict_arg2idx.items()):
                            srl_tokenized_span = [list_tokens[idx] for idx in list_token_idx]
                            srl_span = " ".join(srl_tokenized_span) 
                            dict_triple[arg] = srl_span
                        dict_srl_triples[triple_idx] = dict_triple
                    dict_sent_srl_triples[sent_idx] = dict_srl_triples
                dict_doc_sent_srl_triples[doc_idx] = dict_sent_srl_triples
            dict_ins_doc_sent_srl_triples[ins_idx] = dict_doc_sent_srl_triples
        return dict_ins_doc_sent_srl_triples
