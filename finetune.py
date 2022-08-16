import argparse
from operator import mod
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import  accuracy_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, TrainerCallback

from datasets import load_metric

from dataset import  TextClassificationDataset, GPTClassificationCollator
from data_utils import loading_dataset
from utils import random_sampling, get_model_response
from gpt2 import GPT2LMHeadModelManyShot

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        control_copy = deepcopy(control)
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        return control_copy

    #def on_log(self, args, state, control, **kwargs):
    #    control_copy = deepcopy(control)
    #    self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
    #    return control_copy


class TextClassificationTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        label_token_index = inputs.get("label_tokens")
        del inputs["label_tokens"]
        labels = inputs.get("labels")
        del inputs["labels"]
        bs = inputs['attention_mask'].shape[0]
        ctx_kv_expand = []
        for layer in model.context_kv:
            ctx_kv_expand.append([layer[0].unsqueeze(0).expand(bs,-1,-1,-1), layer[1].unsqueeze(0).expand(bs,-1,-1,-1)])
       
        attention_mask = model.context_mask.unsqueeze(0).expand(bs, -1)
        attention_mask = torch.cat([attention_mask,inputs['attention_mask']], dim=-1)
        ori_mask = inputs['attention_mask']
        del inputs['attention_mask']
        outputs = model(**inputs, past_key_values=ctx_kv_expand, attention_mask=attention_mask, use_cache=False)        
        
        logits = outputs.get("logits")
        vs = logits.size(-1)
        target_inds = (ori_mask.sum(-1)-1).unsqueeze(-1).repeat(1, vs).view(bs, 1, vs)
        prediction_logits = logits.gather(1, target_inds).squeeze(1)
        #print(prediction_logits.shape)
        #prediction_logits = logits[:,-1]
        batch_prediction_logits = prediction_logits[:,label_token_index]
        batch_prediction_probs = torch.softmax(batch_prediction_logits, dim=1)
        loss_ce = torch.nn.CrossEntropyLoss()
        loss = loss_ce(batch_prediction_logits, labels)
       
        return (loss, dict(logits=batch_prediction_probs)) if return_outputs else loss
    
def compute_metrics(eval_preds):
    
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

    

def main(model_name, dataset, bs, epochs, output_dir, lr, num_shot, local_rank=-1, deepspeed=None, evaluate_only=None, resume=None, seed=0, pretrain=None):
    # seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import pickle
   
    # load tokenizer and model
    if 'gpt2' in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)   
        model = GPT2LMHeadModelManyShot.from_pretrained(model_name).cuda()
    elif 'neo'  in model_name:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif 'gpt-j' in model_name:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        raise NotImplementedError
    # to batch generation, we pad on the left and mask those positions out.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    model.to(device)

    
    # prepare and load dataset
    params = dict(dataset=dataset, bs=bs, model=model_name, seed=seed, num_shot=num_shot)

    demon_dataset = TextClassificationDataset(params, 'demon', tokenizer)
    train_dataset = TextClassificationDataset(params, 'train', tokenizer, start_position_id=demon_dataset.position_ids[0][-1] + 1)
    eval_dataset = TextClassificationDataset(params, 'test', tokenizer, start_position_id=demon_dataset.position_ids[0][-1] + 1)
    data_collator = GPTClassificationCollator(tokenizer, params)

    training_args = TrainingArguments(
        output_dir=output_dir, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=epochs, # number of training epochs
        per_device_train_batch_size=bs, # batch size for training
        per_device_eval_batch_size=bs,  # batch size for evaluation
        gradient_accumulation_steps=4,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "steps",
        logging_steps = 20,
        learning_rate=lr,
        weight_decay=0.1,
        warmup_ratio=0.1,
        fp16=True,
        local_rank=local_rank,
        deepspeed=deepspeed,
        remove_unused_columns=False)

    trainer = TextClassificationTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.predict(test_dataset=demon_dataset, metric_key_prefix="demon")
    trainer.evaluate()
    assert False
    if evaluate_only is not None:
        print("Evaluate only with checkpoint {ckpt}.".format(ckpt=evaluate_only))
        if evaluate_only != "":
            trainer._load_from_checkpoint(evaluate_only)
        trainer.evaluate()
        return
    if pretrain is not None:
        trainer._load_from_checkpoint(pretrain)
    #for name, param in model.named_parameters():
    #        print(name, param.shape,param)
    #trainer.add_callback(CustomCallback(trainer)) 
    trainer.train(resume_from_checkpoint=resume)
   

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', dest='model_name', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--dataset', dest='dataset', action='store', required=True, help='name of dataset(s), e.g., agnews')

    # other arguments
    parser.add_argument('--output_dir', dest='output_dir', action='store', required=False, type=str, default='logs')
    parser.add_argument('--local_rank', dest='local_rank', action='store', required=False, type=int, default=-1)
    parser.add_argument('--deepspeed', dest='deepspeed', action='store', required=False, type=str, default=None)

    parser.add_argument('--epochs', dest='epochs', action='store', required=False, type=int, default=10)
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=1)
    parser.add_argument('--lr', dest='lr', action='store', required=False, type=float, default=3e-5)
    parser.add_argument('--evaluate_only', dest='evaluate_only', action='store', required=False, type=str)
    parser.add_argument('--resume', dest='resume', action='store', required=False, type=str)
    parser.add_argument('--pretrain', dest='pretrain', action='store', required=False, type=str)
    parser.add_argument('--seed', dest='seed', action='store', required=False, type=int, default=4)
    parser.add_argument('--num_shot', dest='num_shot', action='store', required=False, type=int, default=4)
    args = parser.parse_args()
    args = vars(args)
    main(**args)