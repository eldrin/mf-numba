import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ...metrics import AveragePrecision
from ..base import TopKRecommender, predict_k
from ...validation import evaluate


def update(Ctr, W, H, reg):
    reg_w, reg_h = reg, reg
    n_users = Ctr.shape[0]
    n_items = Ctr.shape[1]
 
    # update user factors
    data, indices, indptr = Ctr.data, Ctr.indices, Ctr.indptr
    update_factors(n_users, data, indices, indptr, W, H, reg)
    
    # update item factors
    CtrT = Ctr.T.tocsr()
    data, indices, indptr = CtrT.data, CtrT.indices, CtrT.indptr
    update_factors(n_items, data, indices, indptr, H, W, reg)
    
    
@nb.njit("void(i8,f4[:],i4[:],i4[:],f4[:, :],f4[:, :],f4)",
         nogil=True, parallel=True, fastmath=True)
def update_factors(n_entities, data, indices, indptr, X, Y, reg):
    """""" 
    # pre-calc covariance
    YY = Y.dot(Y.T)
    I = np.eye(Y.shape[0]).astype(Y.dtype)
    
    for n in nb.prange(n_entities):
        # get index / values
        i0, i1 = indptr[n], indptr[n+1]
        if i1 - i0 == 0:
            continue
        i_c, v_c = indices[i0:i1], data[i0:i1]
        
        Yc = Y[:, i_c]
        YCmIY = Yc.dot(np.diag(v_c)).dot(Yc.T)
        A = YY + YCmIY + reg * I
        b = Yc.dot(v_c + 1).astype(np.float32)
        
        # update
        X[:, n] = np.linalg.solve(A, b)


def linear_transform(Rtr, alpha, *params):
    """"""
    Ctr = Rtr.copy() 
    Ctr.data = alpha * Ctr.data
    return Ctr.astype(np.float32)


def log_transform(Rtr, alpha, eps, *params):
    """"""
    assert eps > 0
    
    Ctr = Rtr.copy()
    Ctr.data = alpha * np.log(1. + Ctr.data / float(eps))
    return Ctr.astype(np.float32)

        
class WMF(TopKRecommender):
    """"""
    def __init__(self, n_factors, alpha=1., eps=1., reg=0.001,
                 init=0.01, n_epochs=15, transform=linear_transform,
                 dtype=np.float32, verbose=0, monitors=[AveragePrecision()],
                 report_every=1, name='WMF', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.alpha = alpha
        self.eps = eps
        self.reg = reg
        self.init = init
        self.transform = transform
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

        # apply nonlinearity to Rtr
        Ctr = self.transform(Rtr, self.alpha, self.eps)
        
        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Ctr, Rts, progress)
        else:
            self._fit(Ctr, Rts)
            
        return self.user_factors
                       
    def _fit(self, Ctr, Rts=None, progress=None):
        """"""
        for n in range(self.n_epochs):
            update(Ctr, self.user_factors, self.item_factors, reg=self.reg)

            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Ctr, Rts))
            
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