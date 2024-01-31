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

class BertTrainer(BaseTrainer):

    def calculate_batch(self, batch):

        for key, value in batch.items():
            batch[key] = value.to(self.device)

        output = self.model(**batch)
        loss = output.loss
        logits = output.logits

        labels_pred = torch.argmax(logits, dim=2)

        labels_true = batch['labels'].detach().cpu().numpy().tolist()
        labels_pred = labels_pred.detach().cpu().numpy().tolist()

        metric = calc_f1(labels_true, labels_pred)

        output = {
            'loss': loss,
            'metric': metric,
            'logits': logits,
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
