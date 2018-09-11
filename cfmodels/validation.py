import numba as nb
import numpy as np
import pandas as pd
from .metrics import PointwiseMetric, RankingMetric


def split_inner(data, ratio=0.8):
    """ split data (triplet) randomly """
    idx = np.random.permutation(data.shape[0])
    n_train = int(len(idx) * ratio)
    train = data.iloc[idx[:n_train]]
    test = data.iloc[idx[n_train:]]
    return train, test


def split_outer(data, target_column=1, ratio=0.8):
    """ split data (triplet) based on specified entity """    
    unique_entities = data.iloc[:, target_column].unique()
    n_train = int(len(unique_entities) * ratio)
    np.random.shuffle(unique_entities)  # shuffle
    
    # pick training entities
    train_entities = unique_entities[:n_train]
    test_entities = unique_entities[n_train:]

    # split
    train = data[data.iloc[:, target_column].isin(set(train_entities))]
    test = data[data.iloc[:, target_column].isin(set(test_entities))]
    
    return train, test


# currently dict is not supported by numba
# could use cffi based c hash table such as:
# https://github.com/synapticarbors/khash_numba
# but didn't since it's not elengant in terms of installation
@nb.jit
def evaluate(metric, model, user_item, user_item_test, features=None):
    """"""
    n_users = user_item.shape[0]
    n_user_test_items = np.ediff1d(user_item_test.indptr)
    n_user_train_items = np.ediff1d(user_item.indptr)
    
    # aliasing
    Rtr = user_item
    Rts = user_item_test

    # pre-processing the metric cutoffs
    if isinstance(metric, RankingMetric):
        # considering the case where all the (relavant) 
        # train item included in the prediction
        if metric.cutoff is not None:
            # pred_cutoffs = [
            #     metric.cutoff + n_items
            #     if pred_cutoff < user_item.shape[1]
            #     else user_item.shape[1] - 1
            #     for n_items
            #     in n_user_train_items
            # ]
            pred_cutoffs = np.array(n_user_train_items) + metric.cutoff
            pred_cutoffs = np.minimum(pred_cutoffs, metric.cutoff)
            # pred_cutoff = metric.cutoff + n_user_train_items
            # if pred_cutoff >= user_item.shape[1]:
            #     pred_cutoff = user_item.shape[1] - 1
        else:
            # using entire items
            pred_cutoffs = np.ones(user_item.shape[0]) * (user_item.shape[1] - 1)
            # pred_cutoffs = [user_item.shape[1] - 1] * user_item.shape[0]
            metric.cutoff = pred_cutoffs[0]
    
    scores_ = []
    for u in range(n_users):
        if n_user_test_items[u] == 0:
            continue
            
        if isinstance(metric, RankingMetric):
           
            if features is not None:
                pred = model.predict(u, cutoff=pred_cutoffs[u],
                                     features=features)
            else:
                pred = model.predict(u, cutoff=pred_cutoffs[u])

            pred_ = []
            for i in range(len(pred)):
                if len(pred_) >= metric.cutoff:
                    break
            
                if n_user_train_items[u] > 0:
                    train_items = Rtr.indices[Rtr.indptr[u]:Rtr.indptr[u+1]]
                    if np.all(pred[i] != train_items):
                        pred_.append(pred[i])
                else:
                    pred_.append(pred[i])
            pred = np.array(pred_[:metric.cutoff])
            test_items = Rts.indices[Rts.indptr[u]:Rts.indptr[u+1]]
            true = np.array(test_items)
            scores_.append(metric(true, pred))
            
        elif isinstance(metric, PointwiseMetric):
            
            test_ind = Rts.indices[Rts.indptr[u]:Rts.indptr[u+1]]
            true = np.array(Rts.data[Rts.indptr[u]:Rts.indptr[u+1]])
            
            if features is not None:
                pred = model.predict(u, test_ind, features=features)
            else:
                pred = model.predict(u, test_ind)

            scores_.append(metric(true, pred))
            
    return np.mean(scores_)


@nb.jit
def evaluate_fm(metric, model, Xt, yt):
    """"""
    score = 0.
    preds = np.empty((yt.shape[0],), dtype=np.float32)
    if isinstance(metric, RankingMetric):
        # TODO: for BPRFM or Variants
        pass

    elif isinstance(metric, PointwiseMetric):
        for i in nb.prange(Xt.shape[0]):
            preds[i] = model.predict(Xt[i])
        score = metric(yt, preds)
        
    return score