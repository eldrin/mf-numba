import pandas as pd
from os.path import abspath, dirname, join
from .utils import read_data, df2csr, df2fm
from .validation import split_inner, split_time, split_user


def load_test_data():
    data, raw = read_data(join(dirname(__file__), '..', 'data/lastfm55.subset.triplet'), 
                          columns=['user', 'item', 'value'])
    # data, raw = read_data(join(dirname(__file__), '..', 'data/Jam.subset.no_blacklist_mel_safe.triplet'),
    #                       columns=['user', 'item', 'value'])
    # data, raw = read_data(join(dirname(__file__), '..', 'data/ml-100k.csv'),
    #                       columns=['user', 'item', 'value', 'time'])
    # data, raw = read_data('/Users/jaykim/Documents/project/RecSys18Spotify/data/subsets/0/playlist_track_ss_train.csv', 
    #                       columns=['user', 'item', 'value'])
    # datat, rawt = read_data('/Users/jaykim/Documents/project/RecSys18Spotify/data/subsets/0/playlist_track_ss_test.csv', 
    #                       columns=['user', 'item', 'value'], shape=data.shape)
    # return raw, rawt, data, datat
    
    # train, test = split_inner(raw)
    # train, test = split_time(raw)
    train, test = split_user(raw, ratio=0.6)
    Rtr = df2csr(train, data.shape)
    Rts = df2csr(test, data.shape)
    return train, test, Rtr, Rts


def test_bpr(Rtr, Rts, k=20, verbose=1):
    from .factorization import BPR
    from .metrics import AveragePrecision, NDCG, Recall

    bpr = BPR(n_factors=k, lr=0.0001, reg=0.001, init=0.01,
              n_epochs=100, verbose=1,
              report_every=None, monitors=[AveragePrecision(k=10), NDCG(k=10), Recall(k=10)])
    bpr.fit(Rtr, Rts)
    print(bpr.score(Rtr, Rts))
    return bpr
    
    
def test_sgns(Rtr, Rts, k=20, verbose=1):
    from .factorization import SGNS
    sgns = SGNS(n_factors=k, n_negs=5, lr=0.0005, reg=0.1, init=0.01,
                n_epochs=100, verbose=1, report_every=None)
    sgns.fit(Rtr, Rts)
    print(sgns.score(Rtr, Rts))
    return sgns
    
    
def test_wmf(Rtr, Rts, k=20, verbose=1):
    from .factorization import WMF
    from .metrics import AveragePrecision, NDCG, Recall
    wmf = WMF(n_factors=k, reg=0.1, init=0.01, alpha=0.5,
              n_epochs=15, verbose=1, report_every=None,
              monitors=[AveragePrecision(k=10), NDCG(k=10), Recall(k=10)])
    wmf.fit(Rtr, Rts)
    print(wmf.score(Rtr, Rts))
    return wmf
    

def test_implicit_als(Rtr, Rts, k=20):
    from implicit.als import AlternatingLeastSquares
    als = AlternatingLeastSquares(k)
    als.fit(Rtr.T)
    return als
    
    
def test_implicit_bpr(Rtr, Rts, k=20):
    from implicit.bpr import BayesianPersonalizedRanking
    bpr = BayesianPersonalizedRanking(k)
    bpr.fit(Rtr.T)
    return bpr

    
def test_pmf(Rtr, Rts, k=20, verbose=1):
    from .factorization import PMF
    pmf = PMF(n_factors=k, lr=0.002, reg=0.002, init=0.01,
              n_epochs=200, verbose=1, report_every=10)
    pmf.fit(Rtr, Rts)
    # print([{'rmse':v['mse']**0.5} for v in pmf.valid_])
    print({'rmse':pmf.score(Rtr, Rts)['mse']**0.5})
    return pmf
    
    
def test_svdlike(Rtr, Rts, k=20, verbose=1):
    from .factorization import SVDlike as SVD
    svd = SVD(n_factors=k, lr=0.001, reg=0.002, init=0.01,
              n_epochs=100, verbose=1, report_every=10)
    svd.fit(Rtr, Rts)
    # print([{'rmse':v['mse']**0.5} for v in svd.valid_])
    print({'rmse':svd.score(Rtr, Rts)['mse']**0.5})
    return svd


def test_fm(train, test, type='regression', k=20, verbose=1):
    from .factorization import FM
    X, y = df2fm(pd.concat([train, test], axis=0))
    Xt, yt = X[train.shape[0]:], y[train.shape[0]:]  # test set
    X, y = X[:train.shape[0]], y[:train.shape[0]]  # train set

    fm = FM(n_factors=k, type=type, lr=0.0005, reg=0.002, init=0.01,
            n_epochs=100, verbose=1, report_every=None)
    fm.fit(X, y, Xt, yt)
    # print([{'rmse':v['mse']**0.5} for v in fm.valid_])
    print({'rmse':fm.score(Xt, yt)['mse']**0.5})
    return fm


def test_ncf(Rtr, Rts, type='rank', k=20, arch=(20,), verbose=1):
    if type == 'rank':
        from .factorization import NCFRank as NCF
    elif type == 'rating':
        from .factorization import NCFRating as NCF
    
    ncf = NCF(n_factors=k, arch=arch,
              entities={'user':Rtr.shape[0], 'item':Rtr.shape[1]},
              lr=0.01, reg=0.001,
              n_epochs=100, verbose=1, cutoff=500,
              report_every=None)
    ncf.fit(Rtr, Rts)
    print(ncf.score(Rtr, Rts))
    return ncf
    
if __name__ == "__main__":
    
    train, test, Rtr, Rts = load_test_data()
    # model = test_wmf(Rtr, Rts)
    model = test_bpr(Rtr, Rts)
    # model = test_sgns(Rtr, Rts)
    # model = test_implicit_als(Rtr, Rts)
    # model = test_implicit_bpr(Rtr, Rts)
    # model = test_svdlike(Rtr, Rts)
    # model = test_pmf(Rtr, Rts)
    # model = test_fm(train, test)
    # model = test_ncf(Rtr, Rts)