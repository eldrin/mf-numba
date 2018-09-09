import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ..sampling import uniform_sample
from ..metrics import MSE
from ..math import sigmoid
from .base import RatingRecommender, predict
from ..validation import evaluate


@nb.njit(nogil=True, parallel=True)
def update(N, data, indices, indptr, W, H, Wv, Hv, reg, learn_rate, momentum):
    reg_w, reg_h, reg_b = reg, reg, reg
    n_users = len(indptr) - 1
    n_items = np.max(indices)
    users = np.arange(n_users)
    items = np.arange(n_items)
    errsum = 0
    skipped = 0.
    for n in nb.prange(N):
        # sample u, i, j
        sample = uniform_sample(data, indices, indptr, users, items, n_negs=0)
        if sample is None:
            skipped += 1
            continue
        else:
            u, i, v = sample
            
        # update
        tmp_wu = W[:, u].copy()
        wu, hi = tmp_wu, H[:, i]
        wvu, hvi = Wv[:, u], Hv[:, i]

        x_ui = 0.
        for r in range(len(hi)):
            x_ui += wu[r] * hi[r]
        p_ui = sigmoid(x_ui)
        e_ui = v - p_ui
        z_ui = e_ui * p_ui * (1 - p_ui)

        for r in range(len(hi)):
            dwu = -z_ui * hi[r] + reg_w * wu[r]
            Wv[r, u] = momentum * wvu[r] + learn_rate * dwu
            
            dhi = -z_ui * wu[r] + reg_h * hi[r]
            Hv[r, i] = momentum * hvi[r] + learn_rate * dhi
            
            W[r, u] -= Wv[r, u]
            H[r, i] -= Hv[r, i]

        errsum += e_ui**2  # MSE

    return errsum, skipped


class PMF(RatingRecommender):
    """"""
    def __init__(self, n_factors, lr=0.005, momentum=0.9, reg=0.001,
                 init=0.01, n_epochs=100, dtype=np.float32, verbose=0,
                 monitors=[MSE()], report_every=50, name='PMF',
                 *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.lr = lr
        self.momentum = momentum
        self.reg = reg
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
        
        self.user_momentum = np.zeros(
            (self.n_factors, Rtr.shape[0]), dtype=self.dtype)
        self.item_momentum = np.zeros(
            (self.n_factors, Rtr.shape[1]), dtype=self.dtype)
        
        # normalize
        Rtr.data = self._normalize(Rtr.data)

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
            errsum, skipped = update(
                Rtr.nnz, Rtr.data, Rtr.indices, Rtr.indptr,
                self.user_factors, self.item_factors,
                self.user_momentum, self.item_momentum,
                reg=self.reg, learn_rate=self.lr,
                momentum=self.momentum)

            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Rtr, Rts))
            
            if progress is not None:
                progress.update(1)
                progress.set_postfix({
                    'errsum': '{:.4f}'.format(errsum / Rtr.nnz),
                    'skipped': '{:.2%}'.format(skipped / Rtr.nnz)
                })
                
    def predict(self, user, items):
        """"""
        pred = sigmoid(predict(
            user, items, self.user_factors, self.item_factors
        ))
        return self._unnormalize(pred)

    def score(self, Rtr, Rts):
        """"""
        return {
            metric.name: evaluate(metric, self, Rtr, Rts)
            for metric in self.monitors
        }
    
    def _normalize(self, arr):
        """"""
        self.min_ = min(arr)
        self.max_ = max(arr)
        return (arr - self.min_) / (self.max_ - self.min_)

    def _unnormalize(self, arr):
        """"""
        if not hasattr(self, 'min_') or not hasattr(self, 'max_'):
            raise ValueError('[ERROR] Model is not initiated yet!') 
            
        return arr * (self.max_ - self.min_) + self.min_