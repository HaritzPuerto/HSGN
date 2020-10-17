#!/usr/bin/env python
# coding: utf-8
# %%
from tqdm import tqdm
import stanza


class NER_stanza():
    def __init__(self):
        stanza.download('en')  # download English model
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

    def extract_named_entities(self, hotpot, dict_ins2dict_doc2pred):
        list_hotpot_ner = []
        try:
            for instance_idx, hotpot_instance in enumerate(tqdm(hotpot)):
                list_doc_ner = []
                for doc_idx, (doc_title, doc) in enumerate(hotpot_instance['context']):
                    list_sent_ner = []
                    if dict_ins2dict_doc2pred[instance_idx][doc_idx] == 1:
                        for sent_idx, sentence in enumerate(doc):
                            list_sent_ner.append(self.get_ner(sentence))
                    list_doc_ner.append(list_sent_ner)
                list_hotpot_ner.append(list_doc_ner)
            return list_hotpot_ner
        except:
            print("error TT")
            return list_hotpot_ner



# %%
import json
with open('train_top4_doc_ret.json', 'r') as f:
    dict_ins2dict_doc2pred = json.load(f)
with open('../../data/external/hotpot_train_v1.1.json', 'r') as f:
    hotpot = json.load(f)

def golden_docs(hotpot, ins_idx):
    list_sup_facts = hotpot[ins_idx]['supporting_facts']
    set_golden_docs = set([title for title, _ in list_sup_facts])
    title2idx = {title: i for i, (title, _) in enumerate(hotpot[ins_idx]['context'])}
    golden_docs_idx = set()
    for title in set_golden_docs:
        golden_docs_idx.add(str(title2idx[title]))
    return golden_docs_idx


# %%
dict_ins2dict_doc2pred_no_golden_doc = dict()
for ins, dict_doc2pred in dict_ins2dict_doc2pred.items():
    dict_doc2pred_no_golden_doc = dict()
    list_golden = golden_docs(hotpot, int(ins))
    cnt = 0
    for doc, pred in dict_doc2pred.items():
        if doc in list_golden or cnt == 2:
            dict_doc2pred_no_golden_doc[int(doc)] = 0
        else:
            dict_doc2pred_no_golden_doc[int(doc)] = pred
            if pred == 1:
                cnt += 1
    dict_ins2dict_doc2pred_no_golden_doc[int(ins)] = dict_doc2pred_no_golden_doc

# %%
ner = NER_stanza()

list_hotpot_ner = ner.extract_named_entities(hotpot, dict_ins2dict_doc2pred_no_golden_doc)

# %%
with open('train_entities_non_golden.json', 'w+') as f:
    json.dump(list_hotpot_ner, f)

# %%
with open('train_entities_non_golden.json', 'r+') as f:
    a = json.load(f)

# %%
len(a)

# %%
