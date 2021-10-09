# -*- coding: utf8 -*-
#
import math
import random
from collections import defaultdict
from typing import Optional, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from src.model import BiaffineNerModel
from src.model.metrics import Metrics
from src.transform import Transform1DataSet, get_labels
from src.utils import logger


class BiaffineNer(object):
    def __init__(self):
        self.model: Optional[BiaffineNerModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def build_model(self, transformer: str, sequence_length: int, n_labels: int):
        # 更改大些效果会更好
        model = BiaffineNerModel(
            transformer=transformer,
            sequence_length=300,
            n_labels=n_labels
        )
        self.model = model
        return model

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction='none')

    def build_optimizer(
            self, transformer_lr, transformer_weight_decay,
            num_warmup_steps, num_training_steps,
            pretrained: torch.nn.Module,
            lr=1e-5, weight_decay=0.01,
            no_decay=('bias', 'LayerNorm.bias', 'LayerNorm.weight'),

    ):
        if transformer_lr is None:
            transformer_lr = lr
        if transformer_weight_decay is None:
            transformer_weight_decay = weight_decay
        params = defaultdict(lambda: defaultdict(list))
        pretrained = set(pretrained.parameters())
        if isinstance(no_decay, tuple):
            def no_decay_fn(name):
                return any(nd in name for nd in no_decay)
        else:
            assert callable(no_decay), 'no_decay has to be callable or a tuple of str'
            no_decay_fn = no_decay
        for n, p in self.model.named_parameters():
            is_pretrained = 'pretrained' if p in pretrained else 'non_pretrained'
            is_no_decay = 'no_decay' if no_decay_fn(n) else 'decay'
            params[is_pretrained][is_no_decay].append(p)

        grouped_parameters = [
            {'params': params['pretrained']['decay'], 'weight_decay': transformer_weight_decay, 'lr': transformer_lr},
            {'params': params['pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': transformer_lr},
            {'params': params['non_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
            {'params': params['non_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
        ]

        optimizer = AdamW(grouped_parameters, lr=lr, weight_decay=weight_decay, eps=1e-8)

        if num_warmup_steps < 1:
            num_warmup_steps = num_warmup_steps * num_training_steps

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        return optimizer, scheduler

    def build_dataloader(self, file: str, transformer: str, batch_size: int = 32, shuffle=True, max_length=128):
        return Transform1DataSet(
            file=file, transformer=transformer,
            batch_size=batch_size, shuffle=shuffle,
            max_length=max_length,
            device=self.device
        ).to_dataloader()

    def build_metrics(self):
        pass

    @staticmethod
    def set_seed(seed=123321):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # ^^ safe to call this function even if cuda is not available
        torch.cuda.manual_seed_all(seed)

    def get_tokenizer(self, transformer: str):
        tokenizer = AutoTokenizer.from_pretrained(transformer)
        self.tokenizer = tokenizer
        return tokenizer

    def fit(
            self, train_data, dev_data, transformer: str,
            sequence_length=128, epochs=100,
            batch_size: int = 32,
            num_warmup_steps=0.1,
    ):
        self.set_seed()

        train_dataloader = self.build_dataloader(file=train_data, transformer=transformer, batch_size=batch_size,
                                                 max_length=sequence_length,
                                                 shuffle=True)
        dev_dataloader = self.build_dataloader(file=dev_data, transformer=transformer, batch_size=batch_size,
                                               max_length=sequence_length,
                                               shuffle=False)

        model = self.build_model(transformer=transformer, sequence_length=sequence_length,
                                 n_labels=len(get_labels()))
        model.to(self.device)

        criterion = self.build_criterion()
        optimizer, scheduler = self.build_optimizer(
            transformer_lr=1e-4,
            transformer_weight_decay=None,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=len(train_dataloader) * epochs,
            pretrained=model.encoder,

        )

        return self.fit_loop(
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,

        )

    def fit_loop(self, train_dataloader, dev_dataloader, epochs: int, criterion, optimizer, scheduler):
        best_loss, best_f1 = math.inf, 0
        for epoch in range(1, epochs + 1):
            fit_loss = self.fit_dataloader(
                train=train_dataloader, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler
            )
            dev_loss, (precision, recall, f1) = self.evaluate_dataloader(dev=dev_dataloader, criterion=criterion)
            logger.info(
                f'Epoch {epoch},lr: {scheduler.get_last_lr()[0]:.4e} train loss: {fit_loss:.4f}, dev loss: {dev_loss:.4f}, dev precision: {precision:.4f}, dev recall: {recall:.4f}, dev f1:{f1:.4f}'
            )

            if dev_loss < best_loss:
                best_loss = dev_loss
                self.save_weights(save_path='savepoints/old_loss.pt')
            if best_f1 < f1:
                best_f1 = f1
                self.save_weights(save_path='savepoints/old_f1.pt')

    def fit_dataloader(self, train, criterion, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train):
            input_ids, label_ids, mask = batch
            y_pred = self.model(
                input_ids=input_ids,
            )
            loss = self.compute_loss(
                criterion=criterion,
                y_pred=y_pred,
                y_true=label_ids,
                mask=mask
            )
            total_loss += loss.item()
            loss.backward()

            self.step(optimizer=optimizer, scheduler=scheduler)
        return total_loss

    def step(self, optimizer, scheduler):
        #
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def compute_loss(self, criterion, y_pred, y_true, mask):

        y_true = y_true.view(-1)
        y_pred = y_pred.view((-1, y_pred.shape[-1]))
        loss = criterion(input=y_pred, target=y_true)

        mask = mask.view(-1)

        loss *= mask

        avg_loss = torch.sum(loss) / mask.size(0)
        return avg_loss

    @torch.no_grad()
    def evaluate_dataloader(self, dev, criterion):
        self.model.eval()
        metrics = Metrics()
        total_loss = 0
        for batch in tqdm(dev):
            input_ids, label_ids, mask = batch
            y_pred = self.model(
                input_ids=input_ids,
            )
            loss = self.compute_loss(
                criterion=criterion,
                y_pred=y_pred,
                y_true=label_ids,
                mask=mask
            )
            total_loss += loss.item()
            metrics.step(y_true=label_ids, y_pred=y_pred, mask=mask)

        return total_loss, metrics.summary()

    def save_weights(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_weights(self, save_path):
        self.model.load_state_dict(torch.load(save_path))

    @torch.no_grad()
    def predict(self, transformer: str, sequence_length: int, file):
        label_ids, id_labels = None, None
        if self.model is None:
            label_ids = get_labels()
            id_labels = {v: k for k, v in label_ids.items()}
            self.build_model(transformer=transformer, sequence_length=sequence_length, n_labels=len(label_ids) + 1)
            self.load_weights(save_path='savepoints/old_f1.pt')

            self.model.to(self.device)
            self.model.eval()
            self.get_tokenizer(transformer=transformer)

        # just use dev file.
        dataloader = self.build_dataloader(
            file=file,
            transformer=transformer,
            batch_size=1,
            shuffle=False,
            max_length=sequence_length
        )

        results = []
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, mask, label_mask = batch

            y_pred = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

            y_pred = y_pred.argmax(-1).to('cpu')

            for batch_index in range(y_pred.size(0)):
                single_result = []

                y_pred_indices = np.argwhere(y_pred[batch_index])

                for i in range(y_pred_indices.shape[-1]):
                    start, end = y_pred_indices[:, i]
                    argument: List[str] = self.tokenizer.convert_ids_to_tokens(input_ids[batch_index][start:end])
                    argument_type = id_labels.get(int(y_pred[batch_index][start, end]))
                    single_result.append({'type': argument_type, 'argument': argument})
                results.append(single_result)
        return results
