from transformers import BertConfig

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

config = BertConfig(
    vocab_size=28996,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    pad_token_id=0,
    position_embedding_type='absolute',
    use_cache=True,
    classifier_dropout=None,
    id2label=id2tag,
    label2id=tag2id,
)

model = AutoForTokenClassification(config)

model_path = 'model_path'
teacher_model = AutoModelForTokenClassification.from_pretrained(model_path)

lr = 3e-4
factor = 1
max_grad_norm = 10
temperature = 5
n_epochs = 15

model = model.to(device)
teacher_model = teacher_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_steps_warmup = round(1 * len(dataloader_train))
n_steps_training = len(dataloader_train) * n_epochs
scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr(factor, n_steps_warmup, n_steps_training))

config = {
    'model': model.__class__.__name__,
    'teacher_model': teacher_model.__class__.__name__,
    'optimizer': optimizer.__class__.__name__,
    'scheduler': scheduler.__class__.__name__,
    'batch_size': batch_size,
    'lr': lr,
    'factor': factor,
    'max_grad_norm': max_grad_norm,
    'temperature': temperature,
    'n_epochs': n_epochs,
    'n_steps_warmup': n_steps_warmup,
    'n_steps_training': n_steps_training,
    'n_parameters': sum([p.numel() for p in model.parameters()]),
}

name = f'{model.__class__.__name__}-distilled'

trainer = DistillationTrainer(
    model=model,
    teacher_model=teacher_model,
    optimizer=optimizer,
    scheduler=scheduler,
    temperature=temperature,
    device=device,
)

trainer.train(
    dataloader_train=dataloader_train,
    dataloader_val=dataloader_val,
    n_epochs=n_epochs,
    project='distillation_ner_model',
    name=name,
    config=config,
    save=True,
)
