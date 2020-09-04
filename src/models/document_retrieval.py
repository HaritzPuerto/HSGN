#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


class DocumentRetrieval():
    def __init__(self, device, model_path, pretrained_weights='bert-base-uncased'):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)
        self.device = device
        if device == 'cuda':
            self.model.cuda()

    def predict_relevant_docs(self, data, batch_size=16):
        dataloader = self.__create_dataloader(data, batch_size)
        preds = self.__run_model(dataloader)
        dict_ins2dict_doc2pred = self.__convert_preds2dict(data, preds)
        return dict_ins2dict_doc2pred

    def __create_dataloader(self, dev_data, batch_size=16):
        testing_classification_data = []
        for ins in dev_data:
            tmp = {
                "_id": ins['_id'],
                "question": ins['question'],
                "doc_list":[],
                "label": []
            }
            for doc in ins['context']:
                full_doc = " ".join([sent for sent in doc[1]])
                tmp["doc_list"].append(full_doc)
            testing_classification_data.append(tmp)

        test_input_ids = []
        test_attention_masks = []
        sample_id = []

        for idx, sam in enumerate(tqdm(testing_classification_data)):
            for doc in sam['doc_list']:
                encoded_dict = self.tokenizer.encode_plus(
                        sam['question'], doc,
                        add_special_tokens=True,
                        max_length=128,
                        pad_to_max_length=True,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                )

                test_input_ids.append(encoded_dict['input_ids'])
                test_attention_masks.append(encoded_dict['attention_mask'])
                sample_id.append(idx)

        test_input_ids = torch.cat(test_input_ids, dim=0)
        test_attention_masks = torch.cat(test_attention_masks, dim=0)
        # Create the DataLoader.
        prediction_data = TensorDataset(test_input_ids, test_attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        return prediction_dataloader

    def __convert_preds2dict(self, hotpot, preds):
        doc_num = 0
        dict_ins2dict_doc2pred = dict()
        for ins_idx, ins in enumerate(tqdm(hotpot)):
            dict_doc2pred = dict()
            for d_idx, doc in enumerate(ins['context']):
                dict_doc2pred[d_idx] = preds[doc_num]
                doc_num += 1
            dict_ins2dict_doc2pred[ins_idx] = dict_doc2pred
        return dict_ins2dict_doc2pred

    def __run_model(self, dataloader):
        # Put model in evaluation mode
        self.model.eval()
        # Tracking variables 
        predictions = []
        # Predict 
        for batch in tqdm(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
              # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None, 
                                     attention_mask=b_input_mask)
            logits = outputs[0]
          # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
          # Store predictions and true labels
            predictions.extend(logits)
        softmax_predictions = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)
        preds = torch.argmax(softmax_predictions, dim=1).numpy()
        return preds
