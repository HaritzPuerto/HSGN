#!/usr/bin/env python
# coding: utf-8
# %%
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor


# # %%
# data_path = '/workspace/ml-workspace/thesis_git/HSGN/data/'
# hotpotqa_path = 'external/'
# intermediate_train_data_path = 'interim/training/'
# intermediate_dev_data_path = 'interim/dev/'

# with open(os.path.join(data_path, hotpotqa_path, "hotpot_train_v1.1.json"), "r") as f:
#     hotpot_train = json.load(f)
# with open(os.path.join(data_path, hotpotqa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
#     hotpot_dev = json.load(f)


# # %%
# device = 'cuda'


# %%
class SRL():
    def __init__(self, device='cuda'):
        self.predictor = None
        if device == 'cuda':
            self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz", cuda_device=0)
        else:
            self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
        
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

    def extract_srl(self, hotpot):
        dict_ins_doc_sent_srl_triples = dict()
        for ins_idx, hotpot_instance in enumerate(tqdm(hotpot)):
            list_supporting_docs = hotpot_instance['supporting_facts']
            # USE DOC RETRIEVAL LATER ON
            dict_supporting_doc_title2sent_idx = self.__get_dict_supporting_doc_title2sent_idx(list_supporting_docs)
            set_supporting_doc_titles = set(dict_supporting_doc_title2sent_idx.keys())

            dict_doc_sent_srl_triples = dict()
            for doc_idx, (doc_title, doc) in enumerate(hotpot_instance['context']):
                if doc_title not in set_supporting_doc_titles:
                    continue
                list_supporting_sent_idx = []
                if doc_title in set_supporting_doc_titles:
                    list_supporting_sent_idx = dict_supporting_doc_title2sent_idx[doc_title]
                ####### Process sentences #######            
                dict_sent_srl_triples = dict()
                for sent_idx, sent in enumerate(doc):
                    in_supporting_sent = sent_idx in list_supporting_sent_idx
                    ##### SRL Level ######
                    #get SRL instance
                    list_dict_arg2idx, list_tokens = self.sentence2srl_args(sent)

                    ######################################## TRIPLES
                    dict_srl_triples = dict()
                    for triple_idx, dict_arg2idx in enumerate(list_dict_arg2idx):
                        list_args = [] # list of all arguments (node idx) in the triple
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

    def __get_dict_supporting_doc_title2sent_idx(self, list_supporting_docs: list) -> dict:
        '''
        MUST BE REMOVED LATER ON
        Returns dict: str (title) -> int (supporting sentence index)
        '''
        dict_doctitle2sentidx = {}
        for (title, idx) in list_supporting_docs:
            if title in dict_doctitle2sentidx:
                dict_doctitle2sentidx[title].append(idx)
            else:
                dict_doctitle2sentidx[title] = [idx]
        return dict_doctitle2sentidx

# # %%
# srl_model = SRL()


# # %%



# # %%
# print("Extracting SRL from questions of the training set")
# dict_ins_query_srl_triples = create_dictionary_srl(hotpot_train)
# print("Saving")
# with open(os.path.join(data_path, intermediate_train_data_path, "dict_ins_query_srl_triples.json"), "w+") as f:
#     json.dump(dict_ins_query_srl_triples, f)

# # %%
# print("Extracting SRL from questions of the dev set")
# dict_ins_query_srl_triples = create_dictionary_srl(hotpot_dev)
# print("Saving")
# with open(os.path.join(data_path, intermediate_dev_data_path, "dict_ins_query_srl_triples.json"), "w+") as f:
#     json.dump(dict_ins_query_srl_triples, f)

# # %%
# print("Done")
