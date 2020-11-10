import spacy
from pprint import pprint
import os
from utils.preprocess_func import load_data, load_emb_vocab, build_vocab, build_embedding, build_data
from utils.preprocess_utils import *

#path_to_glove = 'data/glove.6B.300d.txt'
path_to_glove = 'data/glove.840B.300d.txt'
data_dir = './data/'

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

train_data = load_data(fname=os.path.join(data_dir, 'train-v2.0.json'), train=True)
dev_data = load_data(fname=os.path.join(data_dir, 'dev-v2.0.json'), train=True)
# pprint(dev_data.shape)

emb_vocab = load_emb_vocab(fname=path_to_glove, dim=EMBED_DIM)
# pprint(emb_vocab)

# set batch size according to your RAM usage
vocab, vocab_tag, vocab_ner = build_vocab(train_data + dev_data, emb_vocab, sort_all=True)
#vocab, vocab_tag, vocab_ner = build_vocab(dev_data, emb_vocab, sort_all=True)

emb = build_embedding(fname=path_to_glove, vocab=vocab, dim=EMBED_DIM)

# Make final data and save in data_dir
build_data(train_data, vocab, vocab_tag, vocab_ner, os.path.join(data_dir, f'train_data_{version}.json'),
            is_train=True, NLP=NLP)
build_data(dev_data, vocab, vocab_tag, vocab_ner, os.path.join(data_dir, f'dev_data_{version}.json'),
            is_train=False, NLP=NLP)
