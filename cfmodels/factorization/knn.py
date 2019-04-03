import numpy as np
from scipy import sparse as sp
from ..metrics import MSE, NDCG

from .base import TopKRecommender, RatingRecommender, predict, predict_k
from ..validation import evaluate


class KNNTopK(TopKRecommender):
    """"""
    def __init__(self, entity='user', metric='cosine', name='KNNTopK',
                 monitors=[NDCG(k=10)], *args, **kwargs):
        """"""
        super().__init__(monitors, name)

        self.entity = entity
        self.metric = metric

    def fit_transform(self, Rtr, Rts=None, *params):
        """"""
        # register training set (will be used for prediction)
        self.Rtr_ = Rtr
    
    def predict(self, user, items):
        """"""
        pass


