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
        if cutoff is None:
            self.name = self.name
        elif isinstance(cutoff, int):
            self.name = '{}@{:d}'.format(self.name, self.cutoff)
        elif isinstance(cutoff, float):
            # TODO: pushing warning for users about rounding
            self.name = '{}@{:d}'.format(self.name, int(np.round(self.cutoff)))
        else:
            raise NotImplementedError
            
        
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
        
        if len(actual) == 0:
            return 0

        return len(set(actual).intersection(set(predicted))) / len(actual)


class EntropyBasedNovelty(MetricBase):
    """"""
    def __init__(self, item_popularity, name='ebn'):
        """"""
        super().__init__(name)
        self.item_popularity = item_popularity

    def __call__(self, predicted):
        """"""
        p = np.array([self.item_popularity[i] for i in predicted])
        return -np.sum(p * np.log2(p))


class DiversityInTopN(MetricBase):
    """"""
    def __init__(self, n_items, k=40, name='adiv'):
        """"""
        super().__init__(name=name)
        self.cutoff = k
        self.n_items = n_items

    def __call__(self, all_predicted):
        """"""
        pred_set = set()
        for pred in all_predicted:
            if len(pred) > self.cutoff:
                pred = pred[:self.cutoff]
            pred_set.update(list(pred))
        
        return len(pred_set) / self.n_items


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
    idcg = 0.
    for i, p in enumerate(predicted):
        if np.any(actual == p) and np.all(predicted[:i] != p):
            dcg += 1. / np.log2(i + 2.)
        if i < len(actual):
            idcg += 1. / np.log2(i + 2.)

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
