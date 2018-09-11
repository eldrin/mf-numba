import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ...metrics import AveragePrecision
from ..base import TopKRecommender, predict_k
from ...validation import evaluate
from .wmf import log_transform, linear_transform


def update(Ctr, F, W, H, PHI, reg_phi, reg_wh):
    reg_w, reg_h = reg_wh, reg_wh
    n_users = Ctr.shape[0]
    n_items = Ctr.shape[1]
 
    # update user factors
    data, indices, indptr = Ctr.data, Ctr.indices, Ctr.indptr
    update_factors_U(n_users, data, indices, indptr, W, H, reg_w)
    
    # update item factors
    CtrT = Ctr.T.tocsr()
    data, indices, indptr = CtrT.data, CtrT.indices, CtrT.indptr
    update_factors_V(n_items, data, indices, indptr, H, W, PHI, F, reg_h, reg_phi)
    
    # update factor-feature (PHI)
    update_factors_W(n_items, H, PHI, F, reg_h, reg_phi)
    
    
@nb.njit("void(i8,f4[:],i4[:],i4[:],f4[:, :],f4[:, :],f4)",
         nogil=True, parallel=True, fastmath=True)
def update_factors_U(n_entities, data, indices, indptr, X, Y, reg_w):
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
        A = YY + YCmIY + reg_w * I
        b = Yc.dot(v_c + 1).astype(np.float32)

        # update
        X[:, n] = np.linalg.solve(A, b)
        
        
@nb.njit("void(i8,f4[:],i4[:],i4[:],f4[:, :],f4[:, :],f4[:, :],f4[:, :],f4,f4)",
         nogil=True, parallel=True, fastmath=True)
def update_factors_V(n_entities, data, indices, indptr, X, Y, W, F, reg_h, reg_phi):
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
        f_i = F[:, n]
        YCmIY = Yc.dot(np.diag(v_c)).dot(Yc.T)
        A = YY + YCmIY + reg_h * I
        b = Yc.dot(v_c + 1) + reg_phi * W.dot(f_i)

        # update
        X[:, n] = np.linalg.solve(A, b)


@nb.njit("void(i8,f4[:, :],f4[:, :],f4[:, :],f4,f4)",
         nogil=True, parallel=True, fastmath=True)
def update_factors_W(n_entities, Y, W, F, reg_h, reg_phi):
    """"""
    # No LOOP
    I = np.eye(F.shape[0]).astype(F.dtype)
    B = reg_h * Y.dot(F.T)
    A = reg_h * F.dot(F.T) + reg_phi * I
    # B = Y.dot(F.T)
    # A = F.dot(F.T) + reg_phi * I
    
    W[:, :] = np.linalg.solve(A, B.T).T


class WMFA(TopKRecommender):
    """"""
    def __init__(self, n_factors, alpha=1., eps=1., reg_phi=100, reg_wh=0.001,
                 init=0.01, n_epochs=15, transform=linear_transform,
                 dtype=np.float32, verbose=0, monitors=[AveragePrecision()],
                 report_every=1, name='WMFA', *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        
        self.n_factors = n_factors        
        self.alpha = alpha
        self.eps = eps
        # self.reg = reg
        self.reg_phi = reg_phi
        self.reg_wh = reg_wh
        self.init = init
        self.transform = transform
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype
        
    def fit_transform(self, Rtr, Rts=None):
        """"""    
        self.user_factors = self._init_factors(
            Rtr[0].shape[0], self.n_factors, self.dtype, self.init)        
        self.item_factors = self._init_factors(
            Rtr[0].shape[1], self.n_factors, self.dtype, self.init)
        self.factor_feature = self._init_factors(
            Rtr[1].shape[0], self.n_factors, self.dtype, self.init)

        # apply nonlinearity to Rtr
        Ctr = (self.transform(Rtr[0], self.alpha, self.eps), Rtr[1])
        
        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(Ctr, Rts, progress)
        else:
            self._fit(Ctr, Rts)
            
        return self.user_factors
                       
    def _fit(self, Ctr, Rts=None, progress=None):
        """"""
        # parse inputs
        Ctr_, Xtr = Ctr
        if Rts is not None:
            Rts_, Xts = Rts
        else:
            Rts_, Xts = None, None
        
        for n in range(self.n_epochs):
            update(Ctr_, Xtr,
                   self.user_factors,
                   self.item_factors,
                   self.factor_feature,
                   reg_phi=self.reg_phi,
                   reg_wh=self.reg_wh)

            if (Rts is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Ctr_, Rts_))
            
            if progress is not None:
                progress.update(1)
                
    def predict(self, user, features=None, cutoff=40):
        """"""
        if features is None:
            item_factors = self.item_factors
        else:
            item_factors = self.factor_feature.dot(features)
            
        return predict_k(
            user, self.user_factors, item_factors, cutoff
        )
        


    def score(self, Rtr, Rts, features=None):
        """"""
        return {
            metric.name: evaluate(metric, self, Rtr, Rts, features)
            for metric in self.monitors
        }