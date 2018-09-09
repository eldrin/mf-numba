import numba as nb
import numpy as np

@nb.jit
def uniform_sample(data, indices, indptr, users, items, n_negs):
    """"""
    u = np.random.choice(users)
    i_u0 = indptr[u]
    i_u1 = indptr[u+1]
    n_pos_u = i_u1 - i_u0
    if n_pos_u == 0:
        return None

    ii = np.random.randint(i_u0, i_u1)
    i = indices[ii]
    v = data[ii]
    
    if n_negs == 0:
        return u, i, v
    
    elif n_negs == 1:
        pos = indices[i_u0:i_u1]
        j = np.random.choice(items)
        # check if it's bad
        bad = False
        for jj in range(len(pos)):
            if j == pos[jj]:
                bad = True
                break
        if bad:
            return None
        
        return u, i, j
    
    else:
        return None
    

# @nb.jit("Tuple((i8[:],f4[:]))(i8,f4[:],i8[:],i8[:])")
@nb.jit
def get_row(row_idx, data, indices, indptr):
    """
    For WMF
    """
    i0 = indptr[row_idx]
    i1 = indptr[row_idx + 1]
    n_elm = i1 - i0
    if n_elm == 0:
        return None
    
    return indices[i0:i1], data[i0:i1]