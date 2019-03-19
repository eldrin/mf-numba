import numpy
from ..base import CollaborativeFilteringBase
from ...metrics import MetricBase, PointwiseMetric, RankingMetric
from ...exceptions import MetricModelMismatch


class UserKNN(CollaborativeFilteringBase):
    """"""
    def __init__(self, k, name='UserKNN'):
        """"""
        super().__init__(None, name)
        self.k = k
    
    def fit_transform(self, Rtr, Rts=None, *params):
        """"""
        pass
