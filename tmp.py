import numpy as np
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch.nn as nn
import torch
import pickle
from data_utils import loading_dataset
from utils import random_sampling, construct_prompt
from scipy.special import logsumexp
from itertools import permutations
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"], use_cache=True)
print(outputs.past_key_values[0][0].shape)