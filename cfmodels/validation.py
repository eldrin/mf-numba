import numba as nb
import numpy as np
from scipy import sparse as sp
import pandas as pd
from .metrics import PointwiseMetric, RankingMetric


def split_inner(csr, ratio=0.8):
    """Split data (triplet) randomly 
    
    Inputs:
        csr (sp.csr_matrix): sparse representation of user-item interaction
        ratio (float): ratio for the train / test split
    
    Returns:
        sp.csr_matrix: train matrix
        sp.csr_matrix: test matrix
    """
    def _split_coo(coo_, rnd_idx):
        return sp.coo_matrix(
            (
                coo.data[idx[:n_train]],
                (
                    coo.row[idx[:n_train]],
                    coo.col[idx[:n_train]]
                )
            ),
            shape=coo.shape
        )
    
    idx = np.random.permutation(csr.nnz)
    n_train = int(len(idx) * ratio)
    
    coo = csr.tocoo()
    train = _split_coo(coo, idx[:n_train]).tocsr()
    test = _split_coo(coo, idx[n_train:]).tocsr()
    
    return train, test


def split_outer(csr, target_column=1, ratio=0.8):
    """Split data (triplet) based on specified entity
    
    Inputs:
        csr (sp.csr_matrix): sparse representation of user-item interaction
        target_column (int): indicates which column to be processed
                             (0 (~= user), 1 (~= item), etc.)
        ratio (float): ratio for the train / test split
    
    Returns:
        sp.csr_matrix: train matrix
        sp.csr_matrix: test matrix
    """
    idx = np.random.permutation(csr.shape[0])
    n_train = int(len(idx) * ratio)
    if target_column == 0:
        train = csr[idx[:n_train]]
        test = csr[idx[n_train:]]
    elif target_column == 1:
        train = csr.tocsc()[:, idx[:n_train]].tocsr()
        test = csr.tocsc()[:, idx[n_train:]].tocsr()
    else:
        # TODO: generalize this to further axes
        raise ValueError(
            '[ERROR] currently only supports either user or item based split!'
        ) 
    
#     unique_entities = data.iloc[:, target_column].unique()
#     n_train = int(len(unique_entities) * ratio)
#     np.random.shuffle(unique_entities)  # shuffle

#     # pick training entities
#     train_entities = unique_entities[:n_train]
#     test_entities = unique_entities[n_train:]

#     # split
#     train = data[data.iloc[:, target_column].isin(set(train_entities))]
#     test = data[data.iloc[:, target_column].isin(set(test_entities))]

    return train, test


def split_time(data, ratio=0.8):
    """"""
    assert data.shape[-1] >= 3  # assuming the last column is time
    tt = np.percentile(data.iloc[:, -1], ratio)
    train = data[data.iloc[:, -1] < tt].iloc[:, :3]
    test = data[data.iloc[:, -1] >= tt].iloc[:, :3]
    return train, test


def split_user(csr, ratio=0.8):
    """Split data (triplet) per user
    
    Inputs:
        csr (sp.csr_matrix): sparse representation of user-item interaction
        ratio (float): ratio for the train / test split
    
    Returns:
        sp.csr_matrix: train matrix
        sp.csr_matrix: test matrix
    """
    # triplet is not re-indexed (original entities)
    # convert triplets into csr mat
    val_tr, col_tr, row_tr = [], [], []
    val_ts, col_ts, row_ts = [], [], []
    for user in range(csr.shape[0]):
        indptr = slice(csr.indptr[user], csr.indptr[user+1])
        items = csr.indices[indptr]
        values = csr.data[indptr]
        idx = np.random.permutation(len(items))
        n_train = int(len(idx) * ratio) 
        
        val_tr.extend(values[:n_train].tolist())
        col_tr.extend(items[:n_train].tolist())
        row_tr.extend([user] * n_train)
        
        val_ts.extend(values[n_train:].tolist())
        col_ts.extend(items[n_train:].tolist())
        row_ts.extend([user] * (len(idx) - n_train))
        
    train = sp.coo_matrix((val_tr, (row_tr, col_tr)), shape=csr.shape).tocsr()
    test = sp.coo_matrix((val_ts, (row_ts, col_ts)), shape=csr.shape).tocsr()     
    
#     train = []
#     test = []
#     for (u, items), (_, value) in zip(
#             data.groupby('user')['item'].apply(list).items(),
#             data.groupby('user')['value'].apply(list).items()):

#         # np.random.shuffle(items)
#         rnd_idx = np.random.permutation(len(items))
#         items = [items[j] for j in rnd_idx]
#         value = [value[j] for j in rnd_idx]
#         bound = int(ratio * len(items))
#         train.extend([(u, i, v) for i, v in zip(items[:bound], value[:bound])])
#         test.extend([(u, i, v) for i, v in zip(items[bound:], value[bound:])])
#     train = pd.DataFrame(train, columns=['user', 'item', 'value'])
#     test = pd.DataFrame(test, columns=['user', 'item', 'value'])

    return train, test


# currently dict is not supported by numba
# could use cffi based c hash table such as:
# https://github.com/synapticarbors/khash_numba
# but didn't since it's not elengant in terms of installation
@nb.jit
def evaluate(metric, model, user_item, user_item_test, features=None):
    """"""
    n_user_test_items = np.ediff1d(user_item_test.indptr)
    n_user_train_items = np.ediff1d(user_item.indptr)

    # aliasing
    Rtr = user_item
    Rts = user_item_test

    # pre-processing the metric cutoffs
    pred_cutoffs = 0
    if isinstance(metric, RankingMetric):
        # considering the case where all the (relavant)
        # train item included in the prediction
        if metric.cutoff is not None:
            pred_cutoffs = np.array(n_user_train_items) + metric.cutoff
            pred_cutoffs = np.minimum(pred_cutoffs, metric.cutoff).astype(int)
        else:
            # using entire items
            pred_cutoffs = np.ones(user_item.shape[0]) * (user_item.shape[1] - 1)
            pred_cutoffs = pred_cutoffs.astype(int)
            metric.cutoff = pred_cutoffs[0]

    if isinstance(metric, RankingMetric):
        return _evaluate_ranking(Rtr, Rts, model, metric, pred_cutoffs,
                                 n_user_train_items, n_user_test_items,
                                 features)

    elif isinstance(metric, PointwiseMetric):
        return _evaluate_rating(Rts, model, metric, n_user_test_items, features)


# @nb.jit
def _evaluate_ranking(Rtr, Rts, model, metric, pred_cutoffs,
                      n_user_train_items, n_user_test_items,
                      features=None):
    """"""
    scores_ = []
    # for u in nb.prange(Rts.shape[0]):
    for u in range(Rts.shape[0]):
        train_items = set(Rtr.indices[Rtr.indptr[u]:Rtr.indptr[u+1]])

        if n_user_test_items[u] == 0:
            continue

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
                # if np.all(pred[i] != train_items):
                if pred[i] not in train_items:
                    pred_.append(pred[i])
            else:
                pred_.append(pred[i])
        pred = np.array(pred_[:metric.cutoff])
        test_items = Rts.indices[Rts.indptr[u]:Rts.indptr[u+1]]
        true = np.array(test_items)
        scores_.append(metric(true, pred))
    return np.mean(scores_)


@nb.jit
def _evaluate_rating(Rts, model, metric, n_user_test_items, features=None):
    """"""
    scores_ = []
    for u in nb.prange(Rts.shape[0]):
        if n_user_test_items[u] == 0:
            continue

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
