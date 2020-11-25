train_data, dev_data = load_data(path_to_data)
glove = load_glove_embed(path_to_glove)

# Create vocabulary
vocab = Vocab() # defined in author’s code
for word in spacy.tokenize(train_data + dev_data): # tokenize done using spacy directly gives tags as well
    vocab.add(word.text)
    vocab.add_tag(word.tag)
    vocab.add_ner(word.named_entity))

# Create Embedding matrix
emb = np.zeros(data_size, 300) # 300 dimensional GLoVE vectors
for word in spacy.tokenize(train_data + dev_data):
    emb.add(glove[word])

# Create pre-processed datasets
questions, passages, raw_ans = process(train_data, vocab, emb)
ans = build_span(passages, raw_ans)
features = get_features(questions) # has PoS, NeR, hard features, etc

# Save everything for later use
save(vocab); save(voab_tags); save(vocab_ner)
save(emb)
save(preproc_train_data); save(preproc_dev_data)
