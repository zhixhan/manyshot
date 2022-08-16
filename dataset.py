from copy import deepcopy
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import os
from utils import construct_prompt, get_model_response, random_sampling
from data_utils import loading_dataset
from sentence_transformers import util

def random_sampling_w_label(params, train_sentences, train_labels):
    train_labels = np.asarray(train_labels)
    sampled_text, sampled_label =[], []
    for ind in range(len(params['label_dict'])):
        index = np.where(train_labels==ind)[0]
        index = np.random.choice(index,params['num_shot'], replace=False).tolist()
        for i in index:
            sampled_text.append(train_sentences[i])
            sampled_label.append(train_labels[i])
    return sampled_text, sampled_label

class TextClassificationDataset(Dataset):
    def __init__(self, params, data_type, tokenizer, start_position_id=0):
        # define variables    
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = loading_dataset(params)
        np.random.seed(params['seed'])
        all_train_sentences, all_train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shot'])
        #np.random.seed(0)
        #all_test_sentences, all_test_labels = random_sampling(all_test_sentences, all_test_labels, 300)
        all_texts = all_test_sentences if data_type=='test' else all_train_sentences
        all_labels = all_test_labels if data_type=='test' else all_train_labels
        if data_type == 'demon':
            all_texts = [construct_prompt(params, [text], [label], "") for text,label in zip(all_texts, all_labels)]
            tokenizer.padding_side = "left"
        else:
            all_texts = [construct_prompt(params, [], [], text) for text in all_texts]
            tokenizer.padding_side = "right"
        
        inputs = tokenizer.batch_encode_plus(all_texts, return_tensors="pt", truncation=True, padding=True)
        inputs['position_ids'] = torch.ones_like(inputs['attention_mask']).long().cumsum(-1) - 1 + start_position_id
        self.attention_mask = inputs['attention_mask']
        self.position_ids = inputs['position_ids']
        self.input_ids = inputs['input_ids']
        self.labels = all_labels
        self.data_type = data_type

    def __len__(self):  
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if self.data_type == 'demon':
            return dict(input_ids=self.input_ids[idx],  attention_mask=self.attention_mask[idx], position_ids=self.position_ids[idx])
        return dict(input_ids=self.input_ids[idx],  attention_mask=self.attention_mask[idx], position_ids=self.position_ids[idx],label=self.labels[idx]) 
       

class DemonClassificationDataset(Dataset):
    def __init__(self, params, data_type, tokenizer, start_position_id=0):
        # define variables    
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = loading_dataset(params)
        np.random.seed(params['seed'])
        all_train_sentences, all_train_labels = random_sampling( all_train_sentences, all_train_labels, params['num_shot'])
        #np.random.seed(0)
        #all_test_sentences, all_test_labels = random_sampling(all_test_sentences, all_test_labels, 2)
        all_texts = all_test_sentences if data_type=='test' else all_train_sentences
        all_labels = all_test_labels if data_type=='test' else all_train_labels
        
        
        all_texts = [construct_prompt(params, all_train_sentences, all_train_labels, text) for text in all_texts]
        tokenizer.padding_side = "left"
        inputs = tokenizer.batch_encode_plus(all_texts, return_tensors="pt", truncation=True, padding=True)
        inputs['position_ids'] = inputs['attention_mask'].long().cumsum(-1) - 1
        inputs['position_ids'].masked_fill_(inputs['attention_mask'] == 0, 1)
        self.attention_mask = inputs['attention_mask']
        self.position_ids = inputs['position_ids']
        self.input_ids = inputs['input_ids']
        self.labels = all_labels
        self.data_type = data_type
        print(inputs)
    def __len__(self):  
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx],  attention_mask=self.attention_mask[idx], position_ids=self.position_ids[idx],label=self.labels[idx]) 

class GPTClassificationCollator(object):

    def __init__(self, tokenizer, params, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.tokenizer = tokenizer
        self.params = params
        # Check max sequence length.
        self.max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
       

    def __call__(self, sequences):
       
        # Get all texts from sequences list.
        input_ids = torch.stack([sequence['input_ids'] for sequence in sequences])
        attention_mask = torch.stack([sequence['attention_mask'] for sequence in sequences])
        position_ids = torch.stack([sequence['position_ids'] for sequence in sequences])
        # Get all labels from sequences list.
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if 'label' in sequences[0]:
            labels = [sequence['label'] for sequence in sequences] if 'label' in sequences[0] else None
            # Encode all labels using label encoder.

            label_token_index = []
            label_dict= self.params['label_dict']
            for label_id, label_list in label_dict.items():
                assert len(label_list)==1
                label = label_list[0]
                label = " " + label
                token = self.tokenizer.encode(label)[0]
                label_token_index.append(token)
            
            inputs.update({'labels': torch.tensor(labels)})
            inputs.update({'label_tokens': label_token_index})
    
        return inputs

'''
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# to batch generation, we pad on the left and mask those positions out.
gpt2_tokenizer.padding_side = "left"
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
params = dict(dataset='sst2')

TokenizedDataset(params,gpt2_tokenizer,training=True)
'''