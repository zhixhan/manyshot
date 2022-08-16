import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num, max_length=None):
    """randomly sample subset of the training pairs"""
    if max_length is not None:
        filtered_sentences = []
        filtered_labels = []
        for index in range(len(sentences)):
            if len(sentences[index])<= max_length:
                filtered_sentences.append(sentences[index])
                filtered_labels.append(labels[index])
        sentences = filtered_sentences
        labels = filtered_labels

    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    

    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

gpt2_model = None
gpt2_tokenizer = None
def setup_gpt2(model_name):
    # load the GPT-2 model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-2 model")
        print('=================')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_model.eval().cuda()
        
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")


def setup_gptj(model_name):
    # load the GPT-j model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-j-6B model")
        gpt2_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def setup_neo(model_name):
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-neo-2.7B model")
        gpt2_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def setup_neox(model_name):
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-neox-20b model")
        gpt2_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"
    if test_sentence != "":
        prompt += q_prefix
        prompt += test_sentence + "\n"
        assert a_prefix[-1] == ' '
        prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    
    
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentences, normalize=True):
    
    all_raw_answers = []
    all_logits = []
    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    prompts = []
    for test_sentence in test_sentences:
        prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        with torch.no_grad():
            if 'gpt2' in params['model']:
                setup_gpt2(params['model'])
            elif 'gptj' in params['model']:
                setup_gptj(params['model'])
            elif 'gptneo' in params['model']:
                setup_neo(params['model'])
            elif 'gpt-neox' in params['model']:
                setup_neox(params['model'])
            logits, resp = complete_gpt2_simple(test_chunk_prompts, params['label_dict'], normalize=normalize)
        for answer_id, answer in enumerate(resp):
            all_raw_answers.append(answer)
        for logit in logits:
            all_logits.append(logit)
    
    return np.asarray(all_logits), np.asarray(all_raw_answers)


def params_check(params):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    if 'gpt2' in params['model']:
        setup_gpt2(params['model'])
    elif 'gptj' in params['model']:
        setup_gptj(params['model'])
    elif 'gptneo' in params['model']:
        setup_neo(params['model'])
    elif 'gpt-neox' in params['model']:
        setup_neox(params['model'])
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            with torch.no_grad():
                input_ids = gpt2_tokenizer.encode(' '+label_name)
                assert len(input_ids)==1, 'label name is more than 1 token'
    
    if not (params['dataset'] in ['rte']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()



#only support one token prediction
def complete_gpt2_simple(prompt, label_dict, l=1, normalize=True):
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    # greedily generate l tokens
    assert l==1
    total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
   
    # we are left padding, so we need to adjust the position IDs
    attention_mask = (total_sequences != 50256).float()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # get the logits for the context and the next l tokens
    logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
    # get the top tokens and probs for the generated l tokens
    prediction_probs = torch.softmax(logits[:,-l-1], dim=1).cpu().numpy()
    prediction_logits = logits[:,-l-1].cpu().numpy()
    #bs x 50257
    num_classes = len(label_dict)
    
    all_test_prediction_logits = []
    all_test_prediction_probs =[]
    for ind in range(prediction_logits.shape[0]):
        label_probs = [0]*num_classes
        label_logits = [0]*num_classes
        for label_id, label_list in label_dict.items():
            assert len(label_list)==1
            label = label_list[0]
            label = " " + label
            token = gpt2_tokenizer.encode(label)[0]
            label_probs[label_id]=prediction_probs[ind][token]
            label_logits[label_id]= prediction_logits[ind][token]
        
        if normalize:
            label_probs = [prob/np.sum(label_probs) for prob in label_probs]
        all_test_prediction_probs.append(label_probs)
        all_test_prediction_logits.append(label_logits)
    
    return all_test_prediction_logits, all_test_prediction_probs

        



