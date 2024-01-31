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


class BaseTrainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device=None,
    ):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.training_epoch = 0
        self.training_step = 0

    def train(
        self,
        dataloader_train,
        dataloader_val=None,
        dataloader_test=None,
        project=None,
        name=None,
        config=None,
        n_epochs=10,
        save=True
    ):

        config = dict() if config is None else config

        with wandb.init(
            project=project,
            settings=wandb.Settings(start_method='fork'),
            config=config,
            name=name,
        ) as run:

            wandb.watch(self.model)

            for epoch in trange(n_epochs, desc=f'Training Model for {n_epochs} Epochs'):

                self.training_epoch += 1

                history = {
                    'loss': [],
                    'metric': [],
                    'labels_true': [],
                    'labels_pred': [],
                }

                self.model.train()
                for batch in tqdm(dataloader_train, desc=f'Training @ Epoch {self.training_epoch}'):

                    self.training_step += 1

                    output = self.calculate_batch(batch)

                    loss = output.loss
                    metric = output.metric

                    labels_true = output.labels_true
                    labels_pred = output.labels_pred

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['max_grad_norm'])
                    self.optimizer.step()

                    learning_rate = float(self.scheduler.get_last_lr()[0])
                    self.scheduler.step()

                    history['loss'].append(loss.item())
                    history['metric'].append(metric.item())

                    history['labels_true'].extend(labels_true)
                    history['labels_pred'].extend(labels_pred)

                    loss_train_batch = loss.item()
                    metric_train_batch = metric.item()

                    wandb.log({
                        'loss_train_batch': loss_train_batch,
                        'metric_train_batch': metric_train_batch,
                        'learning_rate': learning_rate,
                    })

                output = self.calculate_epoch(history)

                loss_train_epoch = output.loss
                metric_train_epoch = output.metric

                wandb.log({
                    'loss_train_epoch': loss_train_epoch,
                    'metric_train_epoch': metric_train_epoch,
                })

                if dataloader_val is not None:
                    loss_val, metric_val = self.evaluate(dataloader_val, desc=f'Validation @ Epoch {self.training_epoch}')
                    wandb.log({
                        'loss_val': loss_val,
                        'metric_val': metric_val,
                    })

                if save:
                    self.save_model(self.model)

            if dataloader_test is not None:
                loss_test, metric_test = self.evaluate(dataloader_test, desc=f'Testing @ Epoch {self.training_epoch}')
                wandb.log({
                    'loss_test': loss_test,
                    'metric_test': metric_test,
                })


    def evaluate(self, dataloader, desc=None):

        desc = f'Evaluation @ Epoch {self.training_epoch}' if desc is None else desc

        history = {
            'loss': [],
            'metric': [],
            'labels_true': [],
            'labels_pred': [],
        }

        self.model.eval()
        for batch in tqdm(dataloader, desc=desc):

            output = self.calculate_batch(batch)

            loss = output.loss
            metric = output.metric

            labels_true = output.labels_true
            labels_pred = output.labels_pred

            history['loss'].append(loss.item())
            history['metric'].append(metric.item())

            history['labels_true'].extend(labels_true)
            history['labels_pred'].extend(labels_pred)

        output = self.calculate_epoch(history)

        loss_eval_epoch = output.loss
        metric_eval_epoch = output.metric

        return loss_eval_epoch, metric_eval_epoch


    def calculate_batch(self, batch):
        pass


    def calculate_epoch(self, history):
        pass


    @staticmethod
    def save_model(model):
        filename = f'{model.__class__.__name__}.pt'
        path = os.path.join('data', filename)

        if not os.path.exists('data'):
            os.makedirs('data')

        torch.save(model, path)

        artifact = wandb.Artifact(filename, type='model')
        artifact.add_file(path)
        wandb.log_artifact(artifact)

        #wandb.save(path)
        os.remove(path)


    @staticmethod
    def load_model(artifact_path):

        user, project, model_version = artifact_path.split('/')
        filename, version = model_version.split(':')

        with wandb.init(project=project) as run:
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()

            path = os.path.join(artifact_dir, filename)

            model = torch.load(path)

        return model
