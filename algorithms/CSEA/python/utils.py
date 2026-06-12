import numpy as np

def get_label(pop_obj, boundary):
    n = pop_obj.shape[0]
    label = np.ones((n, 1)) > 0
    for i in range(boundary.shape[0]):
        label[:,0] = np.logical_and(label[:, 0], np.any(
            pop_obj <= boundary[i], axis=1))
    return label.astype(int)


def data_preprocess(pop_dec, label):

    index_1 = np.where(label > 0.5)[0]
    index_0 = np.where(label <= 0.5)[0]
    k1 = np.full((len(index_1),), False)
    k0 = np.full((len(index_0),), False)
    r_index = np.random.permutation(
        len(index_1))[:np.ceil(0.75*len(index_1)).astype(int)]
    k1[r_index] = True
    r_index = np.random.permutation(
        len(index_0))[:np.ceil(0.75*len(index_0)).astype(int)]
    k0[r_index] = True
    k = np.r_[index_1[k1], index_0[k0]]
    train_dec = pop_dec[k]
    train_lab = label[k]
    rest = np.setdiff1d(np.arange(pop_dec.shape[0]), k)
    test_dec = pop_dec[rest]
    test_lab = label[rest]
    return (train_dec, train_lab), (test_dec, test_lab)
