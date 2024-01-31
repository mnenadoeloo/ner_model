checkpoint = 'bert-base-cased'

batch_size = 32

dataloader_train = DataLoader(
    train_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True
)

dataloader_val = DataLoader(
    valid_dataset,
    collate_fn=data_collator,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False
)

def constant_lr(factor):

    def _constant_lr(step):
        return factor

    return _constant_lr


def warmup_lr(factor, n_steps_warmup, n_steps_training):

    def _warmup_lr(step):
        if step < n_steps_warmup:
            _factor = factor * float(step / n_steps_warmup)
        else:
            _factor = factor * max(0, float(n_steps_training - step) / float(max(1, n_steps_training - n_steps_warmup)))

        return _factor

    return _warmup_lr

def calc_f1(labels: List[List[int]], predictions: List[List[int]]):

    text_labels = [[id2tag[l] for l in label if l != -100] for label in labels]
    text_predictions = []

    for i in range(len(text_labels)):
        sample_text_preds = [id2tag[predictions[i][j + 1]] for j in range(len(text_labels[i]))]
        text_predictions.append(sample_text_preds)

    return f1_score(text_labels, text_predictions)

model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', id2label=id2tag, label2id=tag2id).to(device)

lr = 1e-5
factor = 1
max_grad_norm = 10
n_epochs = 10

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = LambdaLR(optimizer, lr_lambda=constant_lr(factor))

config = {
    'model': model.__class__.__name__,
    'optimizer': optimizer.__class__.__name__,
    'scheduler': scheduler.__class__.__name__,
    'batch_size': batch_size,
    'lr': lr,
    'factor': factor,
    'max_grad_norm': max_grad_norm,
    'n_epochs': n_epochs,
    'n_parameters': sum([p.numel() for p in model.parameters()]),
}

name = model.__class__.__name__

trainer = BertTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)

trainer.train(
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    n_epochs=n_epochs,
    project='ner_model',
    name=name,
    config=config,
    save=True,
)
