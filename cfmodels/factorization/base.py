import numpy as np
import numba as nb
from ..metrics import MetricBase, PointwiseMetric, RankingMetric
from ..exceptions import MetricModelMismatch


class CollaborativeFilteringBase(object):
    """"""
    def __init__(self, monitors=None, name='', *params):
        """"""
        self.name = name
        self.monitors = self._verify_metrics(monitors)
        
    def fit(self, Rtr, Rts=None, *params):
        """"""
        self.fit_transform(Rtr, Rts, *params)
        return self
    
    def fit_transform(self, Rtr, Rts=None, *params):
        """"""
        raise NotImplementedError
    
    def predict(self, *params):
        """"""
        raise NotImplementedError
        
    def score(self, *params):
        """"""
        raise NotImplementedError
        
    def _verify_metrics(self, monitors):
        """"""
        if monitors is not None:
            if isinstance(monitors, (list, tuple)):
                for metric in monitors:
                    is_metric_fit(metric, self)
            elif isinstance(monitors, MetricBase):
                is_metric_fit(monitors, self) 
            return monitors
                
    @staticmethod
    def _init_factors(n, r, dtype=np.float32, init=0.01):
        return np.random.randn(r, n).astype(np.float32) * init

    
class TopKRecommender(CollaborativeFilteringBase):
    """"""
    def __init__(self, monitors, name, *params):
        super().__init__(monitors, name, *params)
    

class RatingRecommender(CollaborativeFilteringBase):
    """"""
    def __init__(self, monitors, name, *params):
        super().__init__(monitors, name, *params)

        
def is_metric_fit(metric, model):
    """"""
    if isinstance(metric, RankingMetric):
        if not isinstance(model, TopKRecommender):
            raise MetricModelMismatch
        
    elif isinstance(metric, PointwiseMetric):
        if not isinstance(model, RatingRecommender):
            raise MetricModelMismatch
            
            
@nb.jit
def predict(u, i, W, H):
    """"""
    # scores = np.zeros((len(i),), dtype=H.dtype)
    # for j in nb.prange(len(i)):
    #     for r in range(W.shape[0]):
    #         scores[j] += W[r, u] * H[r, i[j]]
    scores = W[:, u].T.dot(H).ravel()
    return scores


# np.argpartition not supported yet.
# is it faster then to use argsort?
@nb.jit
def predict_k(u, W, H, k):
    """"""
    scores = W[:, u].T.dot(H).ravel()
    ix = np.argpartition(scores, k)[:k]
    return ix[np.argsort(scores[ix])]

