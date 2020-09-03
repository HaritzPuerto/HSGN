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
