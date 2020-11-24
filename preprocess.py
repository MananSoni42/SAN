import spacy
from pprint import pprint
import os
import argparse
from utils.preprocess_func import load_data, load_emb_vocab, build_vocab, build_embedding, build_data
from utils.preprocess_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--glove_path', type=str, default='data/glove.840B.300d.txt',
                    help='Path to stored glove vectors')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='Path to raw dataset, new dataset is written here')
parser.add_argument('--embedding_dim', type=int, default=300,
                    help='Dimensions to use for GLoVE embeddings')

args = parser.parse_args()

path_to_glove = args.glove_path
data_dir = args.data_dir
EMBED_DIM = args.embedding_dim

NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

# Load data from relevant files located in atgs.data_dir directory (data/ by default)
train_data = load_data(fname=os.path.join(data_dir, 'train-v2.0.json'), train=True)
dev_data = load_data(fname=os.path.join(data_dir, 'dev-v2.0.json'), train=True)
# pprint(dev_data.shape)

# Load glove embeddings
emb_vocab = load_emb_vocab(fname=path_to_glove, dim=EMBED_DIM)
# pprint(emb_vocab)

# Build vocabulary consisting of all words in the train and dev sets
# set batch size according to your RAM usage
vocab, vocab_tag, vocab_ner = build_vocab(train_data + dev_data, emb_vocab, sort_all=True)

# Build the word embeddings using glove vectors for our vocabulary
emb = build_embedding(fname=path_to_glove, vocab=vocab, dim=EMBED_DIM)

# Store embeddings, vocab for later use
meta_path = os.join.path(args.data_dir, f'{args.meta}_{version}.pick')
meta = {'vocab': vocab, 'vocab_tag': vocab_tag, 'vocab_ner': vocab_ner, 'embedding': emb}
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

# Make final data and save in data_dir (default: data/)
build_data(train_data, vocab, vocab_tag, vocab_ner, os.path.join(data_dir, f'train_data_{version}.json'),
            is_train=True, NLP=NLP)
build_data(dev_data, vocab, vocab_tag, vocab_ner, os.path.join(data_dir, f'dev_data_{version}.json'),
            is_train=False, NLP=NLP)
