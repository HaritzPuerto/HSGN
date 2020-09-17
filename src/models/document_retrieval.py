#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from sklearn.metrics import accuracy_score


class DocumentRetrieval():
    def __init__(self, device, model_path, pretrained_weights='bert-base-uncased'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
        self.device = device
        if device == 'cuda':
            self.model.cuda()

    def predict_relevant_docs(self, data, batch_size=16, k=2):
        _input = self.__create_input(data)
        (recall2_idx, _, _) = self.__inference(_input)
        dict_ins2dict_doc2pred = self._create_output_dictionary(recall2_idx)
        return dict_ins2dict_doc2pred

    def __create_input(self, dev_data):
        batch_for_each_query = []
        for sam in tqdm(dev_data):
            question = sam['question']
            title_list = []
            context_list = []
            sampled_idx_list = []

            input_ids_list = []
            token_type_list = []
            attention_list = []
            label_list = []

            for doc in sam['context']:
                title_list.append(doc[0])
                context_list.append(" ".join(doc[1]))
            for fact in sam['supporting_facts']:
                sampled_idx = title_list.index(fact[0])
                if sampled_idx in sampled_idx_list:
                    continue
                sampled_idx_list.append(sampled_idx)

            for idx in list(range(len(sam['context']))):
                context = context_list[idx]
                encoded = self.tokenizer.encode_plus(
                                                    question, context,
                                                    add_special_tokens=True,
                                                    max_length=128,
                                                    pad_to_max_length=True,
                                                    return_attention_mask=True,
                                                    truncation=True,
                )
                input_ids_list.append(encoded['input_ids'])
                token_type_list.append(encoded['token_type_ids'])
                attention_list.append(encoded['attention_mask'])
                if idx in sampled_idx_list:
                    label_list.append([0])
                else:
                    label_list.append([1])

            input_ids_tensor = torch.tensor([f for f in input_ids_list], dtype=torch.long)
            token_type_tensor = torch.tensor([f for f in token_type_list], dtype=torch.long)
            attention_tensor = torch.tensor([f for f in attention_list], dtype=torch.long)
            label_tensor = torch.tensor([f for f in label_list], dtype=torch.long)
            
            batch_for_each_query.append((input_ids_tensor,
                                         token_type_tensor,
                                         attention_tensor,
                                         label_tensor))
        return batch_for_each_query

    def __inference(self, batch_for_each_query):
        recall2_idx = []
        recall3_idx = []
        recall4_idx = []
        predictions = []

        total_acc = 0
        recall2 = 0
        recall3 = 0
        recall4 = 0
        for step, batch in tqdm(enumerate(batch_for_each_query)):
            b_input_ids = batch[0].to(self.device)
            b_input_tokens = batch[1].to(self.device)
            b_input_mask = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)
            _, logits = self.model(b_input_ids, 
                                    token_type_ids=b_input_tokens, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
            label_cpu = b_labels.detach().cpu()
            logit_cpu = logits.detach().cpu()
            total_acc += accuracy_score(label_cpu.view(-1).numpy(), torch.argmax(logit_cpu, dim=1).numpy())

            label_cpu_positive_pos = np.where(label_cpu.view(-1) == 0)[0]
            top2_pos = logit_cpu.numpy()[:, 0].argsort()[-2:][::-1]
            top3_pos = logit_cpu.numpy()[:, 0].argsort()[-3:][::-1]
            top4_pos = logit_cpu.numpy()[:, 0].argsort()[-4:][::-1]

            recall2_idx.append((top2_pos, len(label_cpu)))
            recall3_idx.append((top3_pos, len(label_cpu)))
            recall4_idx.append((top4_pos, len(label_cpu)))
            predictions.append(logit_cpu.numpy()[:, 0])

            recall2 += float(len(np.intersect1d(label_cpu_positive_pos, top2_pos))) / len(label_cpu_positive_pos)
            recall3 += float(len(np.intersect1d(label_cpu_positive_pos, top3_pos))) / len(label_cpu_positive_pos)
            recall4 += float(len(np.intersect1d(label_cpu_positive_pos, top4_pos))) / len(label_cpu_positive_pos)
        return (recall2_idx, recall3_idx, recall4_idx)

    def _create_output_dictionary(self, recall2_idx):
        dict_recall2 = {}
        for qid in range(len(recall2_idx)):
            num_context = recall2_idx[qid][1]
            dict_recall2[qid] = dict.fromkeys(range(num_context),0)

            for predidx in recall2_idx[qid][0]:
                dict_recall2[qid][predidx] = 1
        return dict_recall2
       
