import os
import spacy
import logging
import json
import torch
import argparse
import pickle
from pprint import pprint
import numpy as np
from config import set_args
from utils.preprocess_func import load_data, load_emb_vocab, build_vocab, build_embedding, build_data
from utils.preprocess_utils import *
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from my_utils.utils import set_environment
from my_utils.data_utils import predict_squad, gen_name, gen_gold_name, load_squad_v2_label, compute_acc

args = set_args()
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)

def load_ques_context(ques_path, para_path):
    with open(ques_path) as f:
        ques = list(f.readlines())
    with open(para_path) as f:
        para = f.read().replace('\n', ' ')

    ques = [{"question": q.strip('\r\n'), "id": str(i) } for i,q in enumerate(ques)]
    return [{
            "paragraphs": [{
                "qas": ques,
                "context": para,
            }]
    }], ques, para

logger.info('pre-processing questions and paragraphs ...')
path_to_glove = args.glove
path_to_para = args.para_path
path_to_ques = args.ques_path

EMBED_DIM = 300

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

# Load data from relevant files located in atgs.data_dir directory (data/ by default)
data, ques, para = load_ques_context(path_to_ques, path_to_para)
data = load_data(fname='.', train=False, data=data)

meta_path = 'data/meta_v2.pick'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

# Make final data and save in data_dir (default: data/)
build_data(data, meta['vocab'], meta['vocab_tag'], meta['vocab_ner'], './proc_data.json',
            is_train=False, NLP=NLP)

logger.info('Done. Saved in ./proc_data.json')

model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)
set_environment(args.seed, args.cuda)

logger.info('Launching the SAN')
opt = vars(args)
logger.info('Loading data')

version = 'v2' if args.v2_on else 'v1'
gold_version = 'v2.0' if args.v2_on else 'v1.1'

embedding, opt = load_meta(opt, gen_name(args.data_dir, args.meta, version, suffix='pick'))

data = BatchGen('./proc_data.json',
                batch_size=args.batch_size,
                gpu=args.cuda, is_train=False, elmo_on=args.elmo_on)

checkpoint_path = args.checkpoint_path
logger.info(f'path to given checkpoint is {checkpoint_path}')
checkpoint = torch.load(checkpoint_path) if args.cuda else torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['state_dict']

# Set up the model
logger.info('Loading model ...')
model = DocReaderModel(opt, embedding,state_dict)
model.setup_eval_embed(embedding)
logger.info('done')

if args.cuda:
    model.cuda()

# dev eval
logger.info('Predicting ...')
results, labels = predict_squad(model, data, v2_on=args.v2_on)
logger.info('done')

print('\n\n\n------------ Results ------------\n\n')
print(para, '\n')
for q in ques:
    print(f'Ques: {q["question"]}')
    id = q["id"]
    if results[id]:
        print(f'Ans : {results[id]} (confidence = {round(100*labels[id], 3)} %)')
    else:
        print('Ans : Could not find')
    print('\n')
print('\n---------------------------------\n\n')
