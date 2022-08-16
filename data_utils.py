import pandas as pd
import json
import os
import pickle
import numpy as np
from utils import ROOT_DIR
import datasets
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_sst5():
    train_dataset = load_dataset("SetFit/sst5", split="train")
    test_dataset = load_dataset("SetFit/sst5", split="test")
    return train_dataset['text'], train_dataset['label'], test_dataset['text'], test_dataset['label']

def load_mr():
    train_dataset = load_dataset("rotten_tomatoes", split="train")
    test_dataset = load_dataset("rotten_tomatoes", split="test")
    return train_dataset['text'], train_dataset['label'], test_dataset['text'], test_dataset['label']

def load_subj():
    train_dataset = load_dataset("SetFit/subj", split="train")
    test_dataset = load_dataset("SetFit/subj", split="test")
    return train_dataset['text'], train_dataset['label'], test_dataset['text'], test_dataset['label']

def load_emotion():
    train_dataset = load_dataset("emotion", split="train")
    test_dataset = load_dataset("emotion", split="validation")
    return train_dataset['text'], train_dataset['label'], test_dataset['text'], test_dataset['label']

def load_amazon_polarity():
    def process_raw_data_amazon_polarity(titles, contents):
        sentences = []
        for title, content in zip(titles, contents):
            sentences.append(title + '\nReview: ' + content)
        return sentences

    train_dataset = load_dataset("amazon_polarity", split="train")
    test_dataset = load_dataset("amazon_polarity", split="test")
    return process_raw_data_amazon_polarity(train_dataset['title'], train_dataset['content']), train_dataset['label'], process_raw_data_amazon_polarity(test_dataset['title'], test_dataset['content']), test_dataset['label']

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels


def load_dbpedia():

    train_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])
    
    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    
    return train_sentences, train_labels, test_sentences,  test_labels


def load_rte():
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')
    
    return train_questions, train_answers, test_questions, test_answers

def load_mnli():
    train_dataset = load_dataset("glue", 'mnli', split="train")
    test_dataset = load_dataset("glue", 'mnli', split="validation_matched")

    train_questions = []
    train_answers = []
    for p,q,l in zip(train_dataset['premise'],train_dataset['hypothesis'],train_dataset['label']):
        train_questions.append(p + '\n' + 'question: ' + q + ' true, false, or neither?')
        train_answers.append(l)
    
    test_questions = []
    test_answers = []
    for p,q,l in zip(test_dataset['premise'],test_dataset['hypothesis'],test_dataset['label']):
        test_questions.append(p + '\n' + 'question: ' + q + ' true, false, or neither?')
        test_answers.append(l)
    
    
    return train_questions, train_answers, test_questions, test_answers

def generate_embed_mnli(params):
    train_dataset = load_dataset("glue", 'mnli', split="train")
    test_dataset = load_dataset("glue", 'mnli', split="validation_matched")
    
    train_questions = []
    train_answers = []
    for p,q,l in zip(train_dataset['premise'],train_dataset['hypothesis'], train_dataset['label']):
        train_questions.append(p + ' ' + q)
        train_answers.append(l)
    
    test_questions = []
    for p,q,l in zip(test_dataset['premise'],test_dataset['hypothesis'], test_dataset['label']):
        test_questions.append(p + ' ' + q)

    np.random.seed(params['seed'])
    sampled_sent, _ =random_sampling_w_label(params, train_questions, train_answers)
    model = SentenceTransformer('{}-nli-stsb-mean-tokens'.format('roberta-large'))
    model = model.cuda()
    
    embeddings = []
    for sent in tqdm(sampled_sent):
        emb = model.encode(sent)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    np.save(os.path.join('data/MNLI', "{}_sbert-{}.npy".format('mnli', 'roberta-large')), embeddings)

    embeddings = []
    for sent in tqdm(test_questions):
        emb = model.encode(sent)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    np.save(os.path.join('data/MNLI', "{}_sbert-{}-test.npy".format('mnli', 'roberta-large')), embeddings)
    return

def generate_embed_rte(params):
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + ' ' + q)
    
    np.random.seed(params['seed'])
    sampled_sent, _ =random_sampling_w_label(params, train_questions, train_answers)
    model = SentenceTransformer('{}-nli-stsb-mean-tokens'.format('roberta-large'))
    model = model.cuda()
    
    embeddings = []
    for sent in tqdm(sampled_sent):
        emb = model.encode(sent)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    np.save(os.path.join('data/rte', "{}_sbert-{}.npy".format('rte', 'roberta-large')), embeddings)
   
    
    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + ' ' + q)
    
    embeddings = []
    for sent in tqdm(test_questions):
        emb = model.encode(sent)
        embeddings.append(emb)
    embeddings = np.stack(embeddings)
    np.save(os.path.join('data/rte', "{}_sbert-{}-test.npy".format('rte', 'roberta-large')), embeddings)
    return


def loading_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        
    elif params['dataset'] == 'sst5':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst5()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['terrible'], 1: ['bad'], 2: ['okay'], 3: ['good'], 4: ['great']}
        params['inv_label_dict'] = {'terrible': 0, 'bad': 1, 'okay': 2, 'good': 3, 'great':4}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'mr':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mr()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'subj':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_subj()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Input: "
        params["a_prefix"] = "Type: "
        params['label_dict'] = {0: ['objective'], 1: ['subjective']}
        params['inv_label_dict'] = {'objective': 0, 'subjective': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'amazon_polarity':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_amazon_polarity()
        assert len(orig_train_labels)==len(orig_train_sentences)
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Title: "
        params["a_prefix"] = "Is the review positive or negative? "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1


    elif params['dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the news articles into the categories of Politics, Sports, Business, and Science.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Politics'], 1: ['Sports'], 2: ['Business'], 3: ['Science']}
        params['inv_label_dict'] = {'Politics': 0, 'Sports': 1, 'Business': 2, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'emotion':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_emotion()
        params['prompt_prefix'] = "Classify the messages into the categories of sadness, joy, love, anger, fear and surprise.\n\n"
        params["q_prefix"] = "Message: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['sadness'], 1: ['joy'], 2: ['love'], 3: ['anger'], 4: ['fear'], 5: ['surprise']}
        params['inv_label_dict'] = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        #generate_embed_rte(params)
        #assert False

    elif params['dataset'] == 'mnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mnli()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['true'], 1: ['neutral'], 2:['false']}
        params['inv_label_dict'] = {'true': 0, 'neutral': 1, 'false': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        #generate_embed_mnli(params)

    elif params['dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    else:
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels