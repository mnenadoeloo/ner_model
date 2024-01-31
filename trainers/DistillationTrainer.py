import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output
from tqdm.notebook import tqdm, trange
import wandb

import os

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DistillationTrainer(BaseTrainer):

    def __init__(
        self,
        model,
        teacher_model,
        optimizer,
        scheduler=None,
        temperature=5,
        device=None,
    ):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.temperature = temperature

        self.training_epoch = 0
        self.training_step = 0


    def calculate_batch(self, batch):

        for key, value in batch.items():
            batch[key] = value.to(self.device)

        output = self.model(**batch)
        student_loss = output.loss
        student_logits = output.logits

        with torch.no_grad():
            teacher_output = self.teacher_model(**batch)
            teacher_loss = teacher_output.loss
            teacher_logits = teacher_output.logits

        distillation_loss = self.kl_loss(
            input=F.log_softmax(student_logits / self.temperature, dim=2),
            target=F.softmax(teacher_logits / self.temperature, dim=2),
        )

        loss = student_loss + distillation_loss

        labels_pred = torch.argmax(student_logits, dim=2)

        labels_true = batch['labels'].detach().cpu().numpy().tolist()
        labels_pred = labels_pred.detach().cpu().numpy().tolist()

        metric = calc_f1(labels_true, labels_pred)

        output = {
            'loss': loss,
            'metric': metric,
            'logits': student_logits,
            'labels_true': labels_true,
            'labels_pred': labels_pred,
        }

        output = DotDict(output)

        return output


    def calculate_epoch(self, history):

        loss = sum(history['loss']) / len(history['loss'])
        metric = calc_f1(history['labels_true'], history['labels_pred'])

        output = {
            'loss': loss,
            'metric': metric,
        }

        output = DotDict(output)

        return output
