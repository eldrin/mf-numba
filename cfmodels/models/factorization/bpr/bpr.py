import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ....sampling import uniform_sample
from ....metrics import AveragePrecision
from ....math import sigmoid
from ...base import TopKRecommender, predict_k
from ....validation import evaluate


@nb.njit(nogil=True, parallel=True)
def update(N, data, indices, indptr, W, H, reg, learn_rate):
    reg_w, reg_h, reg_b = reg, reg, reg
    n_users = len(indptr) - 1
    n_items = np.max(indices)
    users = np.arange(n_users)
    items = np.arange(n_items)
    correct = 0
    skipped = 0
    for n in nb.prange(N):
        # sample u, i, j
        sample = uniform_sample(data, indices, indptr, users, items, n_negs=1)
        if sample is None:
            skipped += 1
            continue
        else:
            u, i, j = sample
        
        # update
        tmp_wu = W[:, u].copy()
        wu, hi, hj = tmp_wu, H[:, i], H[:, j]
        bi, bj = H[-1, i], H[-1, j]

        x_uij = 0.
        for r in range(len(hi)):
            x_uij += tmp_wu[r] * (hi[r] - hj[r])
        z_uij = sigmoid(-x_uij)

        for r in range(len(hi) - 1):
            W[r, u] -= learn_rate * (-z_uij * (hi[r] - hj[r]) + reg_w * wu[r])
            H[r, i] -= learn_rate * (-z_uij * wu[r] + reg_h * hi[r])
            H[r, j] -= learn_rate * (z_uij * wu[r] + reg_h * hj[r])
        H[-1, i] -= learn_rate * (-z_uij + reg_b * bi)
        H[-1, j] -= learn_rate * (z_uij + reg_b * bj)

        if z_uij < 0.5:
            correct += 1
            
    return correct, skipped


class BPR(TopKRecommender):
    """"""
    def __init__(self, n_factors, lr=0.005, reg=0.001, init=0.01,
                 n_epochs=100, dtype=np.float32, verbose=0,
                 monitors=[AveragePrecision()], report_every=50,
                 name='BPR', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.lr = lr
        self.reg = reg
        self.init = init
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype

    def fit_transform(self, Rtr, Rts=None):
        """"""
        self.user_factors = self._init_factors(
            Rtr.shape[0], self.n_factors + 1, self.dtype, self.init)
        self.user_factors[-1] = 1.
        
        self.item_factors = self._init_factors(
            Rtr.shape[1], self.n_factors + 1, self.dtype, self.init)
        self.item_factors[-1] = 0.  # bias initiated with zeros

        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Rtr, Rts, progress)
        else:
            self._fit(Rtr, Rts)

        return self.user_factors

    def _fit(self, Rtr, Rts=None, progress=None):
        """"""
        for n in range(self.n_epochs):
            correct, skipped = update(Rtr.nnz, Rtr.data, Rtr.indices, Rtr.indptr,
                                      self.user_factors, self.item_factors,
                                      reg=self.reg, learn_rate=self.lr)

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
        return predict_k(
            user, self.user_factors, self.item_factors, cutoff
        )

    def score(self, Rtr, Rts):
        """"""
        return {
            metric.name: evaluate(metric, self, Rtr, Rts)
            for metric in self.monitors
        }