from tqdm import tqdm
from pprint import pprint
import unicodedata
import json
from collections import Counter
import spacy
import re
import numpy as np

from .preprocess_utils import *

def load_data(fname, train):
    '''
    load data from Squad 2.0
    Returns a list of rows, each containing a dictionary with keys:
    uid, context, question, answer, answer_start, answer_end
    '''
    rows = []
    with open(fname, encoding="utf8") as f:
        data = json.load(f)['data']

    print(f'reading from {fname}')
    for article in tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context = f'{context} {END}'
            for q in paragraph['qas']:
                uid, question = q['id'], q['question']
                answers = q.get('answers', [])
                is_impossible = q.get('is_impossible', False)
                label = 1 if is_impossible else 0
                if train:
                    if len(answers) > 0:
                        answer = answers[0]['text']
                        answer_start = answers[0]['answer_start']
                        answer_end = answer_start + len(answer)
                        row = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                else:
                    row = {'uid': uid, 'context': context, 'question': question, 'answer': answers, 'answer_start': -1, 'answer_end':-1}
                rows.append(row)
    return rows

def load_emb_vocab(fname, dim):
    '''
    load the glove embeddings
    Returns a set of words
    '''
    vocab = set()
    with open(fname, encoding='utf-8') as f:
        next(f) # skip header
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-dim]))
            vocab.add(token)
    return vocab

def build_vocab(data, glove_vocab, sort_all, batch_size=4096, threads=24):
    '''
    Returns vocabulary objects for all words, PoS tags and NER tags
    '''
    nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

    # docs
    print('Tokenizing docs')
    docs = [reform_text(row['context']) for row in data]
    doc_tokened = list(nlp.pipe(docs, batch_size=batch_size, n_threads=threads))

    #questions
    print('Tokenizing questions')
    questions = [reform_text(sample['question']) for sample in data]
    questions_tokened = list(nlp.pipe(questions, batch_size=batch_size, n_threads=threads))

    tag_counter = Counter()
    ner_counter = Counter()
    if sort_all:
        counter = Counter()
        merged = doc_tokened + questions_tokened
        print(f'finding tags and name entities (combined)')
        for tokened in tqdm(merged, total=len(data)):
            counter.update([normalize_text(w.text) for w in tokened if len(normalize_text(w.text)) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update([f'{w.ent_type_}_{w.ent_iob_}' for w in tokened])
        vocab = sorted([w for w in counter if w in glove_vocab], key=counter.get, reverse=True)
    else:
        query_counter = Counter()
        doc_counter = Counter()
        print(f'finding tags and name entities (separate)')
        for tokened in tqdm(doc_tokened, total=len(doc_tokened)):
            doc_counter.update([normalize_text(w.text) for w in tokened if len(normalize_text(w.text)) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update([f'{w.ent_type_}_{w.ent_iob_}' for w in tokened])

        for tokened in tqdm(questions_tokened, total=len(questions_tokened)):
            query_counter.update([normalize_text(w.text) for w in tokened if len(normalize_text(w.text)) > 0])
            tag_counter.update([w.tag_ for w in tokened if len(w.text) > 0])
            ner_counter.update([f'{w.ent_type_}_{w.ent_iob_}' for w in tokened])
        counter = query_counter + doc_counter

        # sort query words
        vocab = sorted([w for w in query_counter if w in glove_vocab], key=query_counter.get, reverse=True)
        vocab += sorted([w for w in doc_counter.keys() - query_counter.keys() if w in glove_vocab], key=counter.get, reverse=True)

    tag_vocab, ner_vocab = None, None
    tag_counter = sorted([w for w in tag_counter], key=tag_counter.get, reverse=True)
    ner_counter = sorted([w for w in ner_counter], key=ner_counter.get, reverse=True)
    tag_vocab = Vocabulary.build(tag_counter)
    ner_vocab = Vocabulary.build(ner_counter)
    print(f'POS Tag vocab size: {len(tag_vocab)}')
    print(f'NER Tag vocab size: {len(ner_vocab)}')

    total = sum(counter.values())
    matched = sum(counter[w] for w in vocab)

    print(f'Raw vocab size vs vocab in glove: {len(counter)}/{len(vocab)}')
    print(f'OOV rate: {round(100.0 * (total - matched)/total, 4)} = {(total - matched)}/{total}') # Out of vocab rate
    vocab = Vocabulary.build(vocab)

    print(f'final vocab size: {len(vocab)}')

    return vocab, tag_vocab, ner_vocab

def build_embedding(fname, vocab, dim):
    '''
    Build word embeddings for each word in vocabulary
    Returns a 2-d matrix of size vocab_size x embedding_dim
    '''
    vocab_size = len(vocab)
    emb = np.zeros((vocab_size, dim))
    emb[0] = 0
    with open(fname, encoding='utf-8') as f:
        next(f) # skip header
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-dim]))
            if token in vocab:
                emb[vocab[token]] = [float(v) for v in elems[-dim:]]
    return emb

def postag_func(toks, vocab):
    return [vocab[w.tag_] for w in toks if len(w.text) > 0]

def nertag_func(toks, vocab):
    return [vocab[f'{w.ent_type_}_{w.ent_iob_}'] for w in toks if len(w.text) > 0]

def tok_func(toks, vocab, doc_toks=None):
    return [vocab[w.text] for w in toks if len(w.text) > 0]

def raw_txt_func(toks):
    return [w.text for w in toks if len(w.text) > 0]

def match_func(question, context):
    ''' return exact match (to a question token) for each word in the context '''
    counter = Counter(w.text.lower() for w in context)
    total = sum(counter.values())
    freq = [counter[w.text.lower()] / total for w in context]
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [1 if w in question_word else 0 for w in context]
    match_lower = [1 if w.text.lower() in question_lower else 0 for w in context]
    match_lemma = [1 if (w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma else 0 for w in context]
    features = np.asarray([freq, match_origin, match_lower, match_lemma], dtype=np.float32).T.tolist()
    return features

def build_span(context, answer, context_token, answer_start, answer_end, is_train=True):
    ''' Returns the exact answer span as a tuple '''
    p_str = 0
    p_token = 0
    t_start, t_end, t_span = -1, -1, []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue
        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            return (None, None, [])
        t_span.append((p_str, p_str + token_len))
        if is_train:
            if (p_str <= answer_start and answer_start < p_str + token_len):
                t_start = p_token
            if (p_str < answer_end and answer_end <= p_str + token_len):
                t_end = p_token
        p_str += token_len
        p_token += 1
    if is_train and (t_start == -1 or t_end == -1):
        return (-1, -1, [])
    else:
        return (t_start, t_end, t_span)

def feature_func(sample, query_tokend, doc_tokend, vocab, vocab_tag, vocab_ner, is_train):
    ''' Builds all features (seperately) and returns a dict with all of them '''
    # features
    fea_dict = {}
    fea_dict['uid'] = sample['uid']
    fea_dict['label'] = sample['label']
    fea_dict['query_tok'] = tok_func(query_tokend, vocab)
    fea_dict['query_pos'] = postag_func(query_tokend, vocab_tag)
    fea_dict['query_ner'] = nertag_func(query_tokend, vocab_ner)
    fea_dict['doc_tok'] = tok_func(doc_tokend, vocab)
    fea_dict['doc_pos'] = postag_func(doc_tokend, vocab_tag)
    fea_dict['doc_ner'] = nertag_func(doc_tokend, vocab_ner)
    fea_dict['doc_fea'] = str(match_func(query_tokend, doc_tokend))
    fea_dict['query_fea'] = str(match_func(doc_tokend, query_tokend))
    doc_toks = [t.text for t in doc_tokend if len(t.text) > 0]
    query_toks = [t.text for t in query_tokend if len(t.text) > 0]
    answer_start = sample['answer_start']
    answer_end = sample['answer_end']
    answer = sample['answer']
    fea_dict['doc_ctok'] = doc_toks
    fea_dict['query_ctok'] = query_toks

    start, end, span = build_span(sample['context'], answer, doc_toks, answer_start,
                                    answer_end, is_train=is_train)
    if is_train and (start == -1 or end == -1): return None
    if not is_train:
        fea_dict['context'] = sample['context']
        fea_dict['span'] = span
    fea_dict['start'] = start
    fea_dict['end'] = end
    return fea_dict

def build_data(data, vocab, vocab_tag, vocab_ner, fout, NLP, is_train, batch_size=4096, threads=24):
    '''
    Builds the final dataset (feature dictionary) and writes it to fout
    '''
    print('Tokenize document')
    passages = [reform_text(sample['context']) for sample in data]
    passage_tokened = [doc for doc in NLP.pipe(passages, batch_size=batch_size, n_threads=threads)]

    print('Tokenize question')
    question_list = [reform_text(sample['question']) for sample in data]
    question_tokened = [question for question in NLP.pipe(question_list, batch_size=batch_size, n_threads=threads)]
    dropped_sample = 0

    print(f'Writing to {fout}')
    with open(fout, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(tqdm(data)):
            feat_dict = feature_func(sample, question_tokened[idx], passage_tokened[idx], vocab, vocab_tag, vocab_ner, is_train)
            if feat_dict is not None:
                writer.write(json.dumps(feat_dict) + '\n')
            else:
                dropped_sample += 1

    print(f'dropped {dropped_sample} in total {len(data)}')
