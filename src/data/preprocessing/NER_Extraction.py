#!/usr/bin/env python
# coding: utf-8
# %%
from tqdm import tqdm
import stanza


stanza.download('en')  # download English model


# # %%
# data_path = '/workspace/ml-workspace/thesis_git/HSGN/data/'
# hotpotqa_path = 'external/'
# intermediate_train_data_path = 'interim/training/'
# intermediate_dev_data_path = 'interim/dev/'

# with open(os.path.join(data_path, hotpotqa_path, "hotpot_train_v1.1.json"), "r") as f:
#     hotpot_train = json.load(f)
# with open(os.path.join(data_path, hotpotqa_path, "hotpot_dev_distractor_v1.json"), "r") as f:
#     hotpot_dev = json.load(f)


# %%
class NER_stanza():
    def __init__(self):
        super(NER_stanza).__init__()
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

    def get_ner(self, doc_str):
        ''' 
        Do not use to get SQuAD answers since it lacks answer position
        '''
        doc = self.nlp(doc_str)
        return [ent.text for ent in doc.ents]

    def extract_named_entities_from_query(self, hotpot):
        list_hotpot_ner = []
        for instance_idx, hotpot_instance in enumerate(tqdm(hotpot)):
            query = hotpot_instance['question']
            list_ent = self.get_ner(query)
            list_hotpot_ner.append(list_ent)
        return list_hotpot_ner

    def extract_named_entities(self, hotpot):
        list_hotpot_ner = []
        for instance_idx, hotpot_instance in enumerate(tqdm(hotpot)):
            list_doc_ner = []
            for doc_idx, (doc_title, doc) in enumerate(hotpot_instance['context']):
                list_sent_ner = []
                for sent_idx, sentence in enumerate(doc):
                    list_sent_ner.append(self.get_ner(sentence))
                list_doc_ner.append(list_sent_ner)
            list_hotpot_ner.append(list_doc_ner)
        return list_hotpot_ner


# # %%
# ner = NER_stanza()


# # %%
# def extract_named_entities_from_query(hotpot):
#     list_hotpot_ner = []
#     for instance_idx, hotpot_instance in enumerate(tqdm(hotpot)):
#         query = hotpot_instance['question']
#         list_ent = ner.get_ner(query)
#         list_hotpot_ner.append(list_ent)
#     return list_hotpot_ner


# # %%
# print("Extracting entities from questions of the training set")
# list_ent_query_training = extract_named_entities_from_query(hotpot_train)
# print("Saving")
# with open(os.path.join(data_path, intermediate_train_data_path, "list_ent_query_training.p"), "wb") as f:
#     pickle.dump(list_ent_query_training, f)

# # %%
# print("Extracting entities from questions of the dev set")
# list_ent_query_dev = extract_named_entities_from_query(hotpot_dev)
# print("Saving")
# with open(os.path.join(data_path, intermediate_train_data_path, "list_ent_query_dev.p"), "wb") as f:
#     pickle.dump(list_ent_query_training, f)


# %%
