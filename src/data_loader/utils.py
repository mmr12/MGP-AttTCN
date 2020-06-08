import numpy as np


def reduce_data(X, n_max=None, n_min=None):
    # data = [Y, T, ind_Y, len_T, X, len_X, labels, ids, static, onset_h]
    # data = [0, 1,     2,     3, 4,     5,      6,   7,      8,       9]
    n_removed = 0
    X = [np.asarray(X[i]) for i in range(len(X))]
    if n_max is not None:
        idx_to_reduce = np.where(X[3] >= n_max)  # 7: num_obs_times
        for pat_no in idx_to_reduce[0]:
            # Y, T, ind_K_D, ind_T
            for data_no in range(3):
                X[data_no][pat_no] = X[data_no][pat_no][-n_max:]
            # num_obs
            X[3][pat_no] = n_max
            # X
            X[4][pat_no] = np.arange(start=int(np.min(X[1][pat_no])),
                                     stop=int(np.max(X[1][pat_no])) + 1)
            # num distinct X
            X[5][pat_no] = len(X[4][pat_no])

    if n_min is not None:
        idx_to_keep = X[3] > n_min  # 7: num_obs_times
        for i in range(len(X)):
            X[i] = np.array(X[i])[idx_to_keep]
        n_removed += np.sum(1 - idx_to_keep)
    return X, n_removed


def new_indices(data):
    Y, T, ind_Y, len_T, X, len_X, labels, static, classes, ids = data
    ind_T = np.zeros_like(ind_Y)
    ind_K_D = np.zeros_like(ind_Y)
    for id in range(len(ids)):
        ind_Ti = np.zeros_like(ind_Y[id])
        ind_Yi = np.asarray(ind_Y[id])[:len_T[id]]
        counter = 0
        for feat in range(45):
            new_items = np.where(ind_Yi == feat)[0]
            ind_Ti[counter: counter + len(new_items)] = new_items
            counter += len(new_items)
        ind_T[id, :len(ind_Ti)] = ind_Ti
        ind_K_Di = np.sort(ind_Yi)
        ind_K_D[id, :len(ind_K_Di)] = ind_K_Di

    # data = [Y, T, ind_K_D, ind_T, len_T, X, len_X, labels, static, classes, ids, ind_Y]
    # data = [0, 1,       2,     3,     4, 5,     6,      7,      8,       9,  10,    11]
    data = [Y, T, ind_K_D, ind_T, len_T, X, len_X, labels, static, classes, ids, ind_Y]

    # shuffle
    idx = np.arange(len(classes))
    np.random.shuffle(idx)
    return [np.asarray(data[i])[idx] for i in range(len(data))]


def pad_raw_data(data):
    # data = [Y, T, ind_Y, len_T, X, len_X, labels, ids, static]
    # data = [0, 1,     2,     3, 4,     5,      6,   7,      8]
    dataset_size = len(data[-2])
    max_num_obs = np.max(list(data[3]))
    max_num_X = np.max(list(data[5]))

    Y_padded = np.zeros((dataset_size, max_num_obs))
    T_padded = Y_padded.copy()
    ind_Y_padded = Y_padded.copy()
    X_padded = np.zeros((dataset_size, max_num_X))

    for i in range(dataset_size):
        Y_padded[i, :data[3][i]] = data[0][i]
        T_padded[i, :data[3][i]] = data[1][i]
        ind_Y_padded[i, :data[3][i]] = data[2][i]
        X_padded[i, :data[5][i]] = data[4][i]

    data[0] = Y_padded
    data[1] = T_padded
    data[2] = ind_Y_padded
    data[4] = X_padded
    return data


def remove_column(data, name):
    try:
        columns = data.columns
    except AttributeError:
        return data
    if name in columns:
        data.drop(columns=name, inplace=True)
    return data


def all_horizons(data):
    # data = [Y, T, ind_Y, len_T, X, len_X, labels, ids, onsets, static]
    # data = [0, 1,     2,     3, 4,     5,      6,   7,      8,      9]
    Y_all = data[0].copy()
    T_all = data[1].copy()
    ind_Y_all = data[2].copy()
    num_distinct_Y_all = data[3].copy()
    X_all = data[4].copy()
    num_distinct_X_all = data[5].copy()
    labels_all = data[6].copy()
    icustay_id_all = data[7].copy()
    static_all = data[9].copy()


    classes_all = np.zeros_like(labels_all)
    abs_max_T = np.max(T_all, axis=1)

    # debug
    keep_all = [np.arange(len(labels_all))]

    for horizon in range(1, 7):
        # get new max times
        max_T = data[8] - horizon
        Filter = data[1] > np.broadcast_to(max_T[:, np.newaxis], data[1].shape)
        data[0][Filter] = 0
        data[1][Filter] = 0
        data[2][Filter] = 0
        data[3] = data[3] - np.sum(Filter, axis=1)

        # reduce num outputs
        data[5] = np.ceil(np.max(data[1], axis=1) - data[1][:, 0]).astype(np.int32)
        for i in range(len(data[5])):
            data[4][i, data[5][i]:] = 0

        # drop empty TS
        kept = np.arange(data[4].shape[0])
        to_keep = (data[3] > 0)
        data = [np.asarray(data[i])[to_keep] for i in range(len(data))]

        kept = kept[to_keep]
        abs_max_T = abs_max_T[to_keep]
        classes = np.ones_like(data[7]) * horizon

        # append
        Y_all = np.concatenate((Y_all, data[0]), axis=0)
        T_all = np.concatenate((T_all, data[1]), axis=0)
        ind_Y_all = np.concatenate((ind_Y_all, data[2]), axis=0)
        num_distinct_Y_all = np.concatenate((num_distinct_Y_all, data[3]), axis=0)
        X_all = np.concatenate((X_all, data[4]), axis=0)
        num_distinct_X_all = np.concatenate((num_distinct_X_all, data[5]), axis=0)
        labels_all = np.concatenate((labels_all, data[6]), axis=0)
        icustay_id_all = np.concatenate((icustay_id_all, data[7]), axis=0)
        static_all = np.concatenate((static_all, data[9]), axis=0)
        classes_all = np.concatenate((classes_all, classes), axis=0)
        keep_all.append(kept)
    # data = [Y, T, ind_Y, len_T, X, len_X, labels, static, classes, ids]
    # data = [0, 1,     2,     3, 4,     5,      6,      7,      8,    9]
    data = [Y_all, T_all, ind_Y_all, num_distinct_Y_all, X_all, num_distinct_X_all, labels_all,
            static_all, classes_all, icustay_id_all]

    return data


def separating_and_resampling(data):
    """
    resamples case patients (or controls) to have a balanced dataset
    then separate the two to sample exactly 50 - 50 in a batch
    :param data:
    :return:
    """
    # separating
    labels = data[7]

    idx_control = labels == 0
    idx_case = labels == 1

    # resampling
    if np.sum(idx_control) > np.sum(idx_case):
        idx_case = np.random.choice(np.where(idx_case)[0], np.sum(idx_control), replace=True, p=None)
    elif np.sum(idx_case) > np.sum(idx_control):
        idx_control = np.random.choice(np.where(idx_control)[0], np.sum(idx_case), replace=True, p=None)

    # separating
    control_data = [data[i][idx_control] for i in range(len(data))]
    case_data = [data[i][idx_case] for i in range(len(data))]
    return case_data, control_data
