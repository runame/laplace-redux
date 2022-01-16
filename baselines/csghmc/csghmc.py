import numpy as np
import torch


class CSGHMCTrainer():

    def __init__(self, model, n_cycles, n_samples_per_cycle, n_epochs, initial_lr, num_batch, total_iters, data_size, weight_decay=5e-4, alpha=0.9):
        self.model = model
        self.n_cycles = n_cycles
        self.n_samples_per_cycle = n_samples_per_cycle
        self.n_epochs = n_epochs
        self.epoch_per_cycle = n_epochs // n_cycles
        self.num_batch = num_batch
        self.total_iters = total_iters
        self.data_size = data_size
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.temperature = 1/data_size

        self.initial_lr = initial_lr
        self.lr = initial_lr

    def adjust_lr(self, epoch, batch_idx):
        rcounter = epoch * self.num_batch + batch_idx
        cos_inner = np.pi * (rcounter % (self.total_iters // self.n_cycles))
        cos_inner /= self.total_iters // self.n_cycles
        cos_out = np.cos(cos_inner) + 1

        self.lr = 0.5 * cos_out * self.initial_lr

    def update_params(self, epoch):
        for p in self.model.parameters():
            if not hasattr(p, 'buf'):
                p.buf = torch.zeros(p.size()).cuda()

            d_p = p.grad
            d_p.add_(p, alpha=self.weight_decay)

            buf_new = (1-self.alpha) * p.buf - self.lr * d_p

            if (epoch % self.epoch_per_cycle) + 1 > self.epoch_per_cycle - self.n_samples_per_cycle:
                eps = torch.randn(p.size()).cuda()
                buf_new += (2*self.lr * self.alpha * self.temperature / self.data_size)**.5 * eps

            p.data.add_(buf_new)
            p.buf = buf_new
