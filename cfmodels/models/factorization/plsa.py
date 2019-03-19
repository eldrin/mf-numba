import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ...sampling import uniform_sample
from ...metrics import AveragePrecision
from ...math import sigmoid
from ..base import TopKRecommender, predict_k
from ...validation import evaluate


@nb.njit(nogil=True, parallel=True)
def update(N,
           data, indices, indptr,
           data_t, indices_t, indptr_t,
           W, H):
    # reg_w, reg_h, reg_b = reg, reg, reg
    n_d = len(indptr) - 1
    n_w = np.max(indices)
    Nd = np.sum(data)
    n_factors = W.shape[-1]
    r_dw = np.zeros((n_factors, len(data)))

    for n in range(N):
        # DUMB IMPLEMENTATION
        W_ = W.copy()
        H_ = H.copy()
        W[:] = 0.
        H[:] = 0.
        for row in nb.prange(len(indptr) - 1):
            for i in range(indptr[row], indptr[row+1]):
                d, w, n_d_w = row, indices[i], data[i]
                for k in range(n_factors):
                    W[d, k] += (
                        n_d_w * 
                        (W_[d, k] * H_[w, k]) /
                        np.dot(W_[d, :], H_[w, :]) / Nd
                    )

        for row in nb.prange(len(indptr_t) - 1):
            for i in range(indptr_t[row], indptr_t[row+1]):
                w, d, n_d_w = row, indices_t[i], data_t[i]
                for k in range(n_factors):
                    H[w, k] += (
                        n_d_w * 
                        (W_[d, k] * H_[w, k]) /
                        np.dot(W_[d, :], H_[w, :])
                    )

        denom = np.sum(H, axis=0)
        for w in range(n_w):
            H[w, :] = H[w, :] / denom[w]

        # TODO: SMART VERSION
        # # E-step: get the posterior P(z)P(d|z)P(w|z)/Z
        # for row in nb.prange(len(indptr)-1):
        #     for i in range(indptr[row], indptr[row+1]):
        #         d, w, n_d_w = row, indices[i], data[i]
        #         # TODO: caching denominator
        #         for k in range(n_factors):
        #             r_dw[k, i] = (W[d, k] * H[w, k]) / np.dot(W[d, :], H[w, :])

        # # M-step: adjust the factors to maximize the likelihood
        # # 1. Update W
        # for row in np.prange(len(indptr) - 1):
        #     for i in range(indptr[row], indptr[row+1]):
        #         d, w, n_d_w = row, indices[i], data[i]
        #         for k in range(n_factors):
        #             W[d, k] = n_d_w * r_dw[k, i] / n_d

        # # 2. Update H 


class PLSA(TopKRecommender):
    """"""
    def __init__(self, n_factors, n_epochs=10, dtype=np.float32, verbose=0,
                 monitors=[AveragePrecision()], report_every=50,
                 name='BPR', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype

    @staticmethod
    def _init_factors(n, r, dtype=np.float32):
        """"""
        Q = np.random.rand(n, r).astype(np.float32)
        Q /= Q.sum(axis=1)[:, None]
        return Q

    def fit_transform(self, Rtr, Rts=None):
        """"""
        self.user_factors = self._init_factors(
            Rtr.shape[0], self.n_factors, self.dtype)
        self.item_factors = self._init_factors(
            Rtr.shape[1], self.n_factors, self.dtype)

        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Rtr, Rts, progress)
        else:
            self._fit(Rtr, Rts)

        return self.user_factors

    def _fit(self, Rtr, Rts=None, progress=None):
        """"""
        Rtr_t = Rtr.T.tocsr()
        for n in range(self.n_epochs):
            update(Rtr.nnz,
                   Rtr.data, Rtr.indices, Rtr.indptr,
                   Rtr_t.data, Rtr_t.indices, Rtr_t.indptr,
                   self.user_factors, self.item_factors)

            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Rtr, Rts))
            
            if progress is not None:
                progress.update(1)

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
