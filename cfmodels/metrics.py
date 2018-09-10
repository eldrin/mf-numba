import numpy as np
import numba as nb


class MetricBase(object):
    """"""
    def __init__(self, name):
        """"""
        self.name = name
    
    def __call__(self, actual, predicted, *params):
        """"""
        raise NotImplementedError
    
    
class PointwiseMetric(MetricBase):
    """"""
    def __init__(self, name):
        """"""
        super().__init__(name)

    
class RankingMetric(MetricBase):
    """"""
    def __init__(self, cutoff, name):
        """"""
        super().__init__(name)
        self.cutoff = cutoff
        self.name = '{}@{:d}'.format(self.name, self.cutoff)

        
class AveragePrecision(RankingMetric):
    """"""
    def __init__(self, k=40, name='ap'):
        """"""
        super().__init__(k, name)
    
    def __call__(self, actual, predicted):
        """"""
        return apk(actual, predicted, self.cutoff)
    
    
class MSE(PointwiseMetric):
    """"""
    def __init__(self, name='mse'):
        """"""
        super().__init__(name)
        
    def __call__(self, actual, predicted):
        """"""
        return mse(actual, predicted)


class NDCG(RankingMetric):
    """ NDCG (for binary relavance) """
    def __init__(self, k=40, name='ndcg'):
        """"""
        super().__init__(k, name)
    
    def __call__(self, actual, predicted):
        """"""
        return ndcg(actual, predicted, self.cutoff)
    

class Recall(RankingMetric):
    """"""
    def __init__(self, k=40, name='recall'):
        """"""
        super().__init__(k, name)
    
    def __call__(self, actual, predicted):
        """"""
        if len(predicted) > self.cutoff:
            predicted = predicted[:self.cutoff]
        
        return len(set(actual).intersection(set(predicted))) / len(actual)
    

@nb.jit
def apk(actual, predicted, k=10):
    """ for binary relavance """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.
    num_hits = 0.
    
    for i, p in enumerate(predicted):
        if np.any(actual == p) and np.all(predicted[:i] != p):
            num_hits += 1.0
            score += num_hits / (i + 1.)
    
    if len(actual) == 0:
        return 0.
    
    return score / min(len(actual), k)


@nb.jit
def ndcg(actual, predicted, k=10):
    """ for binary relavance """
    if len(predicted) > k:
        predicted = predicted[:k]

    dcg = 0.
    for i, p in enumerate(predicted):
        if np.any(actual == p) and np.all(predicted[:i] != p):
            dcg += 1. / np.log2((i+1) + 1.)

    idcg = np.sum(1. / np.log2(np.arange(1, len(actual)+1) + 1.))
    
    if len(actual) == 0:
        return 0.
    
    return dcg / idcg


@nb.jit
def mse(actual, predicted):
    """"""
    n = len(actual)
    score = 0.
    for i in range(n):
        score += (actual[i] - predicted[i])**2
    score /= n
    return score