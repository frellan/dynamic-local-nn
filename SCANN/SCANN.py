import os

from SCANN.UnsupervisedModel import UnsupervisedModel
from SCANN.BioClassifier import BioClassifier
from SCANN.BioLoss import BioLoss

from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

experiment = os.path.abspath(os.curdir).split('/')[-1]
tb_logs_dir = f'./data/logs/{experiment}_tb_logs'
tb = SummaryWriter(tb_logs_dir)

class SCANN(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.weights = torch.rand((output_size, input_size), dtype=torch.float).to(self.device)

    def learn_unsupervised_weights(
        self,
        train_loader,
        epochs,
        learning_rate=2e-2,
        precision=1e-30,
        anti_hebbian_learning_strength=0.4,
        lebesgue_norm=2.0,
        rank=2):
        for epoch in range(epochs):
            eps = learning_rate * (1 - epoch / epochs)
            batch_size = iter(train_loader).next()[0].shape[0]

            batch = 1
            for mini_batch, _ in train_loader:
                mini_batch = torch.transpose(mini_batch, 0, 1).to(self.device)
                sign = torch.sign(self.weights)
                W = (sign * torch.abs(self.weights) ** (lebesgue_norm - 1)).to(self.device)
                tot_input = torch.mm(W, mini_batch)

                y = torch.argsort(tot_input, dim=0).to(self.device)
                yl = torch.zeros((self.output_size, batch_size), dtype = torch.float).to(self.device)
                yl[y[self.output_size - 1,:], torch.arange(batch_size)] = 1.0
                yl[y[self.output_size - rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength

                xx = torch.sum(yl * tot_input,1)
                xx = xx.unsqueeze(1)
                xx = xx.repeat(1, self.input_size)
                ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * self.weights

                nc = torch.max(torch.abs(ds))
                if nc < precision: nc = precision
                self.weights += eps * (ds / nc)
                print("Epoch: {}/{}, Batch: {}/{}".format(epoch + 1, epochs, batch, len(train_loader)))
                batch += 1

    def run_test(self, train_dl, test_dl, epochs, lr=1e-3, verbose=0):
        start = time()

        model = BioClassifier(self.weights, 10)
        loss = BioLoss(10)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = create_supervised_trainer(model, optimizer, loss, device=self.device)
        evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss)}, device=self.device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch(trainer):
            evaluator.run(train_dl)
            metrics = evaluator.state.metrics
            tb.add_scalar('train loss', metrics['loss'], trainer.state.epoch)
            tb.add_scalar('train accuracy', metrics['accuracy'], trainer.state.epoch)
            tb.add_scalar('train/loss', metrics['loss'], trainer.state.epoch)
            tb.add_scalar('train/accuracy', metrics['accuracy'], trainer.state.epoch)

            evaluator.run(test_dl)
            metrics = evaluator.state.metrics
            tb.add_scalar('test loss', metrics['loss'], trainer.state.epoch)
            tb.add_scalar('test accuracy', metrics['accuracy'], trainer.state.epoch)
            tb.add_scalar('test/loss', metrics['loss'], trainer.state.epoch)
            tb.add_scalar('test/accuracy', metrics['accuracy'], trainer.state.epoch)

        @trainer.on(Events.COMPLETED)
        def log_complete(engine):
            evaluator.run(test_dl)
            print("Final Accuracy: {:.2f} Took: {:.0f}s".format(evaluator.state.metrics['accuracy'], time() - start))

        trainer.run(train_dl, max_epochs=epochs)