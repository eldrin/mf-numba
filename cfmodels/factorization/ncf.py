import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..sampling import uniform_sample
from ..metrics import AveragePrecision
from ..math import sigmoid
from .base import TopKRecommender, predict_k
from ..validation import evaluate


class SparseDense(nn.Module):
    """"""
    def __init__(self, n_factors, entities, arch, nonlinearity=F.relu):
        """"""
        super().__init__()
        
        self.embedding_layers = nn.ModuleDict(
            {name:nn.Embedding(n_entities, n_factors, sparse=True)
             for name, n_entities in entities.items()}
        )
        
        arch_ = [len(entities) * n_factors]
        arch_.extend(list(arch))
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(arch_[i-1], arch_[i])
             for i in range(1, len(arch_))]
        )
        
        self.nonlinearity = nonlinearity
    
    def forward(self, entity_indices):
        """"""
        # sanity check
        assert (set(entity_indices.keys()) ==
                set(self.embedding_layers.keys()))
        
        # forward-pass for embedding (sparse) layers
        # and concatenation
        x = torch.cat([
            self.embedding_layers[name](i)
            for name, i in entity_indices.items()
        ], dim=-1)
        
        # forward-pass for hidden (dense) layers
        for l in self.hidden_layers:
            x = self.nonlinearity(l(x))
        return x


class NCFRank(TopKRecommender):
    """"""
    def __init__(self, n_factors, entities={'user':1000, 'item':2000},
                 arch=(20,), lr=0.005, reg=0.001, init=0.01,
                 n_epochs=100, dtype=np.float32, verbose=0,
                 monitors=[AveragePrecision()], report_every=50,
                 name='NCFRank', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors
        self.arch = arch
        self.entities = entities
        self.lr = lr
        self.reg = reg
        self.init = init
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype
        
        # initiate model
        self.model = SparseDense(self.n_factors, self.entities, self.arch)
        params = filter(lambda w: w.requires_grad, self.model.parameters())
        self.opt = optim.Adagrad(params, lr=lr)

    def fit_transform(self, Rtr, Rts=None):
        """"""
        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Rtr, Rts, progress)
        else:
            self._fit(Rtr, Rts)

        return self.user_factors

    def _fit(self, Rtr, Rts=None, progress=None):
        """"""
        n_users = len(Rtr.indptr) - 1
        n_items = np.max(Rtr.indices)
        users = np.arange(n_users)
        items = np.arange(n_items)
        
        for n in range(self.n_epochs):
            correct = 0
            skipped = 0

            for n in range(Rtr.nnz):
                # sample u, i, j
                sample = uniform_sample(Rtr.data, Rtr.indices, Rtr.indptr,
                                        users, items, n_negs=1)
                if sample is None:
                    skipped += 1
                    continue
                else:
                    u, i, j = sample
                
                # wrap to tensor
                u = torch.LongTensor([u])
                i = torch.LongTensor([i])
                j = torch.LongTensor([j])
                    
                # update the model
                self.opt.zero_grad()
                pos = self.model({'user':u, 'item':i})
                neg = self.model({'user':u, 'item':j})
                l = -F.logsigmoid(pos - neg).sum()  # BPR
                l.backward()
                self.opt.step()
                
                if torch.exp(-l) > 0.5:
                    correct += 1
                
            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Rtr, Rts))
            
            if progress is not None:
                progress.update(1)
                progress.set_postfix({
                    'correct': '{:.2%}'.format(correct / Rtr.nnz),
                    'skipped': '{:.2%}'.format(skipped / Rtr.nnz)
                })

    def predict(self, user, cutoff=40):
        """"""
        raise NotImplementedError

    def score(self, Rtr, Rts):
        """"""
        raise NotImplementedError