import numba as nb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp

from ...metrics import MSE
from ...math import sigmoid
from ..base import RatingRecommender, predict
from ...validation import evaluate_fm


@nb.njit("f4(i4[:],f4[:],f4[:],f4[:,:])")
def forward(i_x, v_x, w, V):
    """"""
    # add w0
    score = w[0]
    
    # add wx
    for j in range(len(i_x)):
        i = i_x[j]
        x_j = v_x[j]
        
        score += w[i+1] * x_j
     
    # add vvxx
    vf2 = np.zeros((1, V.shape[1]), dtype=V.dtype)
    v2f2 = np.zeros((1, V.shape[1]), dtype=V.dtype)
    for j in range(len(i_x)):
        i = i_x[j]
        x_j = v_x[j]
        
        vi = V[i]
        vf2 += vi * x_j
        v2f2 += (vi**2 * x_j**2)
    vf2 = vf2**2  # (1, f)
 
    for k in range(V.shape[1]):
        score += 0.5 * (vf2[0, k] - v2f2[0, k])
        
    return score


@nb.njit("f4(b1,i8,f4[:],i4[:],i4[:],f4[:],f4[:],f4[:, :],f4,f4)",
         nogil=True, parallel=True)
def update_sgd(is_clf, n_rows, data, indices, indptr, y, w, V, reg, learn_rate):
    """"""
    errsum = 0.
    error = 0.
    for n in nb.prange(n_rows):
        # get index / values
        i0, i1 = indptr[n], indptr[n+1]
        if i1 - i0 == 0:
            continue
        i_x, v_x = indices[i0:i1], data[i0:i1]
        
        y_true = y[n]
        y_pred = forward(i_x, v_x, w, V)
        
        if is_clf:
            error = sigmoid(y_pred * y_true) - 1
        else:
            error = 2. * (y_pred - y_true)
        
        # update
        v_ = np.zeros((V.shape[1],), dtype=V.dtype)
        for j in range(len(i_x)):
            i = i_x[j]
            x = v_x[j]
            
            v_ += V[i] * x  # caching
            
        # updating w0
        w[0] -= learn_rate * (error + 2 * reg * w[0])
        
        # updating w / v
        for j in range(len(i_x)):
            i = i_x[j]
            x = v_x[j]
            
            w[i+1] -= learn_rate * (error * x + 2 * reg * w[i+1])
            for f in range(V.shape[1]):
                V[i, f] -= learn_rate * (error * x * (v_[f] - V[i, f] * x) + 2 * reg * V[i, f])
        
        if is_clf:
            errsum += error
        else:
            errsum += (error/2.)**2
            
    return errsum


class FM(RatingRecommender):
    """"""
    def __init__(self, n_factors, type='regression', lr=0.005, reg=0.001,
                 init=0.01, n_epochs=100, dtype=np.float32, verbose=0,
                 monitors=[MSE()], report_every=10, name='FM',
                 *args, **kwargs):
        """"""
        super().__init__(monitors, name)
        # TODO make a dedicated exception
        assert type in {'regression', 'classification'}
        
        self.n_factors = n_factors
        self.type = type
        self.lr = lr
        self.reg = reg
        self.init = init
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.verbose = verbose
        self.dtype = dtype 

    def fit_transform(self, X, y, Xt=None, yt=None):
        """"""
        X, y = self._check_inputs(X, y)
        Xt, yt = self._check_inputs(Xt, yt)
        
        # w0 is the first element of weight \w
        self.w = np.zeros((X.shape[-1] + 1,), dtype=self.dtype)
        self.V = self._init_factors(
            X.shape[1], self.n_factors, self.dtype, self.init).T       
        
        self.valid_ = []
        if self.verbose > 0:
            with tqdm(total=self.n_epochs, ncols=80) as progress:
                self._fit(X, y, Xt, yt, progress)
        else:
            self._fit(X, y, Xt, yt)
            
        return self.V

    def _fit(self, X, y, Xt=None, yt=None, progress=None):
        """"""
        for n in range(self.n_epochs):
            n_rows = X.shape[0]
            data = X.data
            indices = X.indices
            indptr = X.indptr
            
            errsum = update_sgd(
                True if self.type == 'classification' else False,
                n_rows, data, indices, indptr, y,
                self.w, self.V, self.reg, self.lr)
                
            if (Xt is not None and
                    yt is not None and
                    self.report_every is not None and
                    n % self.report_every == 0):
                self.valid_.append(self.score(Xt, yt))
            
            if progress is not None:
                progress.update(1)
                progress.set_postfix({
                    'errsum': '{:.4f}'.format(errsum / n_rows),
                })
                
    def predict(self, x):
        """
        x (csr_matrix, (1, d)):
            1 row of feature
        """
        i_x = x.indices[x.indptr[0]:x.indptr[1]]
        v_x = x.data[x.indptr[0]:x.indptr[1]]
        return forward(i_x, v_x, self.w, self.V)

    def score(self, Xt, yt):
        """"""
        return {
            metric.name: evaluate_fm(metric, self, Xt, yt)
            for metric in self.monitors
        }

    def _check_inputs(self, X, y):
        """"""
        if X is None or y is None:
            return None, None
        
        # force cast to csr
        X = X.tocsr()
        
        # force casting type to float if it's not
        if self.type == 'classification' and set(y) != {0, 1}:
            # TODO: make Exception
            raise ValueError('[ERROR] only binary classification is supported!')
        else:
            y = y.astype(np.float32)
            
        return X, y