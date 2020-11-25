model = SAN(classifier='on')
train_data = BatchGen(path_to_preprocessed_train_data) # BatchGen defined by author
train_labels = load_labels(path_to_preprocessed_train_data) # Use for evaluating various metrics
dev_data = BatchGen(path_to_preprocessed_dev_data) # BatchGen defined by author
dev_labels = load_labels(path_to_preprocessed_dev_data) # Use for evaluating various metrics

optimizer = Adam(model.parameters())

for epoch in range(epochs):
    loss = 0.0
    train_em, train_f1, train_acc = 0,0,0
    dev_em, dev_f1, dev_acc = 0,0,0

    for batch in train_data:
        y = model(batch['x'])
        loss =  cross_entropy(y['ans_start'], batch['actual_start']) + cross_entropy(y['ans_end'], batch['actual_end'])
        if classifier is 'on':
            loss += binary_cross_entropy(y['label'], batch['actual_label'])
        loss.backward()
        optimizer.step()

        em, f1 = get_em_f1(train_labels, batch, y)
        train_em += (em / len(batch))
        train_f1 += (f1 / len(batch))
        if classifier is 'on':
            train_acc += (y['label']-batch['acutal_label']).sum().item() / len(batch)

save(model)
save(train_em); save(train_f1); save(train_acc)
save(dev_em); save(dev_f1); save(dev_acc)


y_pred = model(dev_data)
em, f1 = get_em_f1(dev_labels, dev_data, y) # dev_data needed for paragraph (context info)
if classifier is 'on':
    acc = (y['label']-dev_labels).sum().item()
    actual_labels = (dev_labels > 0.5)
    predicted_labels = (y_pred['label'] > 0.5)

    cm = confusion_matrix(predicted_labels, actual_labels)) # scipy
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='binary') # scipy
else:
    acc = 0
    cm = None
    precision, recall, f1 = 0,0,0
