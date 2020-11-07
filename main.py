import spacy
from pprint import pprint
import os
import pickle

from preprocess import load_data, load_emb_vocab, build_vocab, build_embedding, build_data
from utils import *

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

train_data = load_data(fname='data/train-v2.0.json', train=True)
dev_data = load_data(fname='data/dev-v2.0.json', train=True)

# pprint(data)
emb_vocab = load_emb_vocab(fname='data/glove.6B.300d.txt', dim=EMBED_DIM)
# pprint(emb_vocab)

# set batch size according to your RAM usage
# This one is too much for my RAM ;-;
#vocab, vt, bner = build_vocab(train_data + dev_data, emb_vocab, sort_all=True)
vocab, vt, bner = build_vocab(dev_data, emb_vocab, sort_all=True)

emb = build_embedding(fname='data/glove.6B.300d.txt', vocab=vocab, dim=EMBED_DIM)

with open('data/vocab_tag.pick','rb') as f:
    vocab_tag = pickle.load(f)
with open('data/vocab_ner.pick','rb') as f:
    vocab_ner = pickle.load(f)

print('---compare---')
print(vocab_tag, vt)
print(vocab_ner, vner)
print('------')

# Make final data and save in data/ directory
build_data(train_data, vocab, vocab_tag, vocab_ner, f'data/train_data_{version}.json',
            is_train=True, NLP=NLP)
build_data(dev_data, vocab, vocab_tag, vocab_ner, f'data/dev_data_{version}.json',
            is_train=False, NLP=NLP)
