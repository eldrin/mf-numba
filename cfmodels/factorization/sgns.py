import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ..metrics import AveragePrecision
from ..math import sigmoid
from .base import TopKRecommender, predict_k
from ..validation import evaluate


@nb.njit(parallel=True, nogil=True)
def update(N, data, indices, indptr, W, H, reg, learn_rate, n_negs):
    reg_w, reg_hi, reg_hj = reg, reg, reg
    n_users = len(indptr) - 1
    n_items = np.max(indices)
    users = np.arange(n_users)
    items = np.arange(n_items)
    correct = 0
    for n in nb.prange(N):
        # sample u, i, j
        u = np.random.choice(users)
        i_u0 = indptr[u]
        i_u1 = indptr[u+1]
        n_pos_u = i_u1 - i_u0
        if n_pos_u == 0:
            return None
        
        pos = indices[i_u0:i_u1]
        # ii = np.random.randint(i_u0, i_u1)
        ii = np.random.choice(pos)
        i = indices[ii]
        v = data[ii]
        
        wu, hi = W[:, u].copy(), H[:, i]        
        
        # TODO: find the way abstract following routine
        #       in numba compatible (and optimized) function
        x = 0.
        for r in range(len(hi)):
            x += wu[r] * hi[r]
        z_ui = 1. / (1. + np.exp(x))

        # update positive sample related factors (only once)
        for r in range(len(hi)):
            W[r, u] -= learn_rate * (-v * z_ui * hi[r] + reg_w * wu[r])
            H[r, i] -= learn_rate * (-v * z_ui * wu[r] + reg_hi * hi[r])
        
        avg_z_uj = 0.
        for m in range(n_negs):
            # draw negative sample
            j = np.random.choice(items)
            bad = False
            for p in pos:
                if j == p:
                    bad = True
                    break

            # make sure we have valid negative item
            while bad:
                j = np.random.choice(items)
                bad = False
                for p in pos:
                    if j == p:
                        bad = True
                        break
                        
            # slice corresponding item factor
            hj = H[:, j]
            
            # TODO: this as well
            x = 0.
            for r in range(len(hj)):
                x += wu[r] * hj[r]
            z_uj = 1. / (1. + np.exp(-x))
            avg_z_uj += z_uj

            for r in range(len(hi)):
                W[r, u] -= learn_rate * (v * z_uj * hj[r])
                H[r, j] -= learn_rate * (v * z_uj * wu[r] + reg_hj * hj[r])
        
        if (1 - z_ui) > avg_z_uj / np.float32(n_negs):
            correct += 1
 
    return correct


class SGNS(TopKRecommender):
    """"""
    def __init__(self, n_factors, lr=0.005, reg=0.001, n_negs=5,
                 init=0.01, n_epochs=100, dtype=np.float32, verbose=0,
                 monitors=[AveragePrecision()], report_every=50,
                 name='SGNS', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.lr = lr
        self.reg = reg
        self.n_negs = n_negs
        self.init = init
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype
        
    def fit_transform(self, Rtr, Rts=None):
        """"""
        self.user_factors = self._init_factors(
            Rtr.shape[0], self.n_factors, self.dtype, self.init)        
        self.item_factors = self._init_factors(
            Rtr.shape[1], self.n_factors, self.dtype, self.init)

        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Rtr, Rts, progress)
        else:
            self._fit(Rtr, Rts)
                       
    def _fit(self, Rtr, Rts=None, progress=None):
        """"""
        for n in range(self.n_epochs):
            correct = update(Rtr.nnz, Rtr.data, Rtr.indices, Rtr.indptr,
                             self.user_factors, self.item_factors,
                             reg=self.reg, n_negs=self.n_negs,
                             learn_rate=self.lr)

            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Rtr, Rts))
            
            if progress is not None:
                progress.update(1)
                progress.set_postfix({
                    'correct': '{:.2%}'.format(correct / Rtr.nnz),
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