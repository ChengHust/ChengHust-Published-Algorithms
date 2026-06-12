
import numpy as np

def is_member(a,b):
    """
    ismember(GLoc,UniqueGLoc,'rows') in Matlab
    """
    return np.nonzero(np.all(b == a[:, np.newaxis], axis=2))[1]

def radar_grid(pop_obj, div):
    n, m = np.shape(pop_obj)
    theta = np.arange(0.0, 2.*np.pi, 2.*np.pi/m)
    r_loc = np.zeros((n, 2))
    r_loc[:, 0] = np.sum(pop_obj * np.cos(theta), axis=1,
                         keepdims=True).T / np.sum(pop_obj, axis=1, keepdims=True).T
    r_loc[:, 1] = np.sum(pop_obj * np.sin(theta), axis=1,
                         keepdims=True).T / np.sum(pop_obj, axis=1, keepdims=True).T
    r_loc = (r_loc + 1) / 2.
    y_lower = np.min(r_loc, axis=0, keepdims=True)
    y_upper = np.max(r_loc, axis=0, keepdims=True)
    if np.any(y_lower == y_upper):
        norm_r = r_loc
    else:
        norm_r = (r_loc - y_lower) / (y_upper-y_lower)
    g_loc = np.floor(norm_r * div)
    g_loc[g_loc >= div] = div - 1
    uni_g = np.unique(g_loc, axis=0)
    site = is_member(g_loc, uni_g)  # g_loc in uni_q
    return site, r_loc
