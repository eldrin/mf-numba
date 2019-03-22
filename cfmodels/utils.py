import pandas as pd
from scipy import sparse as sp
import numpy as np
from tqdm import tqdm


def read_data(fn, columns=['user', 'item', 'value'], shape=None):
    ll = pd.read_csv(fn, header=None, index_col=None, names=columns)
    i, j, v = ll[columns[0]], ll[columns[1]], ll[columns[2]]
    if shape is None:
        shape = (max(i) + 1, max(j) + 1)

    D = sp.coo_matrix((v, (i, j)), shape=shape).tocsr()
    D.eliminate_zeros()
    return D, ll


def df2csr(df, shape=None):
    """"""
    i, j, v = df['user'], df['item'], df['value']
    csr = sp.coo_matrix((v, (i, j)), shape=shape).tocsr()
    csr.eliminate_zeros()
    return csr


def densify(triplet, entity_mins={'user':5, 'item':5}, verbose=False):
    """ Densifying the triplets based on the minimum interaction
    
    Args:
        triplet (pandas.DataFrame): triplet data (only suppurt user-item-value)
        entity_mins (int) : minimum number other objects that are interacted with target entity
        verbose (bool) : boolean flag for the verbosity
    """
    if verbose:
        print('Before filtering: ', triplet.shape)
    j = 0
    d = 1
    data = triplet.copy()
    assert data.shape[-1] == 3
    if any([entity not in data.columns for entity in entity_mins.keys()]):
        raise ValueError(
            '[ERROR] all the requested entity should be included in the data'
        )

    while d > 0:
        d_ = data.shape[0]
        for entity, thres in entity_mins.items():
            n_interactions = data.groupby(entity).size()
            data_ = data[
                data[entity].isin(n_interactions[n_interactions > thres].index)
            ]
        d = d_ - data.shape[0]
        
        if verbose:
            print('Iteration {:d}:'.format(j), data.shape)
            
        j += 1
        
    return data


def df2fm(df):
    """Convert triplet into fm design matrix 
    
    Assuming input triplet is already re-indexed
    Always treats last columns as target \y
    """
    
    # parse target
    y = df.iloc[:, -1].values
    df2 = df.iloc[:, :-1].copy()
    
    # find offsets per feature group
    r = [0]  # numbers of features
    for c in range(df2.shape[-1]):
        r.append(df2.iloc[:, c].nunique())
        
    # apply offsets
    for i in range(len(r) - 1):
        df2.iloc[:, i] = df2.iloc[:, i] + r[i]
    
    # get updated coordinate to build design matrix
    # TODO: current approach is too naive and slow. better way?
    i = 0
    I, J, V = [], [], []
    with tqdm(total=df2.shape[0], ncols=80) as progress:
        for _, row in df2.iterrows():
            for j in row.values:
                I.append(i)
                J.append(j)
                V.append(1) 
            i += 1
            progress.update(1)
            
    # build design matrix
    X = sp.coo_matrix((V, (I, J)), dtype=np.float32).tocsr()
    
    return X, y
