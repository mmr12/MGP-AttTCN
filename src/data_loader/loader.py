import os
import pickle
import sys
import tensorflow as tf
import numpy as np

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print
from src.data_loader.utils import reduce_data, new_indices, pad_raw_data, all_horizons, separating_and_resampling


class DataGenerator:
    def __init__(self,
                 no_mc_samples=10,
                 max_no_dtpts=None,
                 min_no_dtpts=None,
                 batch_size=10,
                 fast_load=False,
                 to_save=False,
                 debug=False,
                 fixed_idx_per_class=False,
                 features=None):
        t_print("DataGenerator -- init")
        cwd = os.path.dirname(os.path.abspath(__file__))
        self.head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
        self.no_mc_samples = no_mc_samples
        self.max_no_dtpts = max_no_dtpts
        self.min_no_dtpts = min_no_dtpts
        self.debug = debug
        """
        Data loader for MIMIC III data preprocessed according to 
        """
        if fast_load:
            self.fast_load(features)
        else:
            self.long_load(to_save, features)

        # data = [Y, T, ind_K_D, ind_T, len_T, X, len_X, labels, static, classes, ids, ind_Y]
        # data = [0, 1,       2,     3,     4, 5,     6,      7,      8,       9,  10,    11]
        if debug == False:
            # remove IDs & debugging cat
            self.train_data = self.train_data[:-2]
            self.val_data = self.val_data[:-2]
            self.test_data = self.test_data[:-2]
        # data = [Y, T, ind_K_D, ind_T, len_T, X, len_X, labels, static, classes]
        # data = [0, 1,       2,     3,     4, 5,     6,      7,      8,       9]
        # separating two prediction classes
        self.train_case_data, self.train_control_data = separating_and_resampling(self.train_data)
        self.len_data = len(self.train_case_data)
        self.train_case_idx = np.arange(len(self.train_case_data[-1]))
        self.train_control_idx = np.arange(len(self.train_control_data[-1]))
        self.val_idx = np.arange(len(self.val_data[-1]))
        # creating a small dev set
        if fixed_idx_per_class:
            self.idx_per_class = np.asarray(
                [[343, 3476, 4378, 1297, 2695, 1498, 1119, 2788, 5468, 5217, 3505,
                  5441, 3895, 4177, 5678, 1108, 5739, 1510, 7, 5055],
                 [5311, 2932, 2091, 6683, 568, 6851, 6273, 2796, 4336, 5342, 3150,
                  1835, 7040, 7106, 3495, 2538, 6053, 2949, 64, 2382],
                 [1976, 2652, 4208, 1472, 3718, 4287, 3972, 2683, 1112, 2083, 3960,
                  5617, 403, 6244, 4370, 886, 3416, 5687, 5226, 6358],
                 [2597, 1086, 6930, 286, 2492, 3794, 21, 1794, 4680, 4477, 6460,
                  6293, 4636, 4788, 5134, 6544, 7139, 2516, 2617, 351],
                 [2812, 1503, 1677, 6553, 6333, 7023, 4310, 5546, 7054, 4522, 4473,
                  1218, 422, 242, 6286, 944, 109, 4896, 3611, 4737],
                 [4837, 3445, 4256, 465, 2720, 7117, 2665, 4109, 590, 5680, 2672,
                  6070, 5697, 3772, 4219, 1298, 6515, 2965, 1788, 3352],
                 [5496, 1159, 3029, 4189, 848, 4778, 2966, 4159, 2101, 6102, 4191,
                  7135, 349, 7003, 483, 4068, 4420, 2885, 2103, 2460]]
            )
        else:
            self.idx_per_class = np.zeros((7, batch_size * 2), dtype=np.int32)
            for k in range(7):
                self.idx_per_class[k] = np.random.choice(np.where(self.val_data[9] == k)[0],
                                                         min(batch_size * 2, len(np.where(self.val_data[9] == k)[0])),
                                                         replace=False, p=None)
        # list of patients present at horizon 6
        # train
        self.late_case_patients = list(self.train_case_data[10][self.train_case_data[9] == 6])
        self.late_control_patients = list(self.train_control_data[10][self.train_control_data[9] == 6])
        self.later_case_patients = list(self.train_case_data[10][self.train_case_data[9] == 6])
        # val
        self.late_val_patients = list(self.val_data[10][self.val_data[9] == 6])
        late_val_pat_id = [self.val_data[10][i] in self.late_val_patients for i in range(len(self.val_data[9]))]
        self.late_val_pat_id = np.where(late_val_pat_id)[0]
        self.horizon0_val_patients = np.where(late_val_pat_id & (self.val_data[9] == 0))[0]

    def apply_reshuffle(self):
        """
        Function linked to training class: the dataset is reshuffled at the beginning of each epoch
        the training class reshuffles the indices, then calls 'apply_reshuffle' to reshuffle the dataset itself
        """
        self.train_case_data = [self.train_case_data[i][self.train_case_idx] for i in range(self.len_data)]
        self.train_control_data = [self.train_control_data[i][self.train_control_idx] for i in range(self.len_data)]
        late_case_pat_id = [self.train_case_data[10][i] in self.late_case_patients
                            for i in range(len(self.train_case_data[9]))]
        late_control_pat_id = [self.train_control_data[10][i] in self.late_control_patients
                               for i in range(len(self.train_control_data[9]))]
        self.late_case_pat_id = np.where(late_case_pat_id)[0]
        self.late_control_pat_id = np.where(late_control_pat_id)[0]
        self.horizon0_case_patients = np.where(late_case_pat_id & (self.train_case_data[9] == 0))[0]
        self.horizon0_control_patients = np.where(late_control_pat_id & (self.train_control_data[9] == 0))[0]

    def next_batch(self, batch_size, batch, loss='uni', alignment=-1, time_window=25, late=False, horizon0=False):
        # first: create new dataset
        if late:
            data = [np.concatenate((self.train_case_data[i][self.late_case_pat_id[batch * batch_size:
                                                                                  (batch + 1) * batch_size]],
                                    self.train_control_data[i][self.late_control_pat_id[batch * batch_size:
                                                                                        (batch + 1) * batch_size]]))
                    for i in range(self.len_data)]
        elif horizon0:
            data = [np.concatenate((self.train_case_data[i][self.horizon0_case_patients[batch * batch_size:
                                                                                        (batch + 1) * batch_size]],
                                    self.train_control_data[i][self.horizon0_control_patients[batch * batch_size:
                                                                                              (
                                                                                                          batch + 1) * batch_size]]))
                    for i in range(self.len_data)]
        else:
            data = [np.concatenate((self.train_case_data[i][batch * batch_size: (batch + 1) * batch_size],
                                    self.train_control_data[i][batch * batch_size: (batch + 1) * batch_size]))
                    for i in range(self.len_data)]

        # then reshuffle it
        idx = np.random.choice(np.arange(len(data[4])), len(data[4]), replace=False)
        data = [data[i][idx] for i in range(self.len_data)]
        output = self.extract_data(data)
        if loss == 'uni':
            yield output
        else:
            output[7] = self.expand_labels(output[7], alignment=alignment, time_window=time_window)
            yield output

    def next_batch_dev_small(self, batch):
        data = [self.val_data[i][self.idx_per_class[batch]] for i in range(len(self.val_data))]
        yield self.extract_data(data)

    def next_batch_dev_all(self, batch_size, batch, late=False, horizon0=False):
        if late:
            data = [self.val_data[i][self.late_val_pat_id[batch * batch_size: (batch + 1) * batch_size]]
                    for i in range(len(self.val_data))]
            """
            elif horizon0:
            
                data = [self.val_data[i][self.horizon0_val_patients[batch * batch_size: (batch + 1) * batch_size]]
                        for i in range(len(self.val_data))]
            """
        else:
            data = [self.val_data[i][batch * batch_size: (batch + 1) * batch_size] for i in range(len(self.val_data))]
        yield self.extract_data(data)

    def next_batch_test_all(self, batch_size, batch, late=False, horizon0=False):
        data = [self.test_data[i][batch * batch_size: (batch + 1) * batch_size] for i in range(len(self.test_data))]
        yield self.extract_data(data)

    def next_batch_train_all(self, batch_size, batch, late=False, horizon0=False):
        data = [self.train_data[i][batch * batch_size: (batch + 1) * batch_size] for i in range(len(self.train_data))]
        yield self.extract_data(data)

    def extract_data(self, data):
        # data = [Y, T, ind_K_D, ind_T, num_distinct_Y, X, num_distinct_X, labels, static, classes]
        # data = [0, 1,       2,     3,              4, 5,              6,      7,      8,       9]
        # second: extract
        # list of datapoints collected per patient
        Y = data[0]
        Y = tf.convert_to_tensor(Y, dtype=tf.float32, name='Y')
        # list of corresponding timestamps
        T = data[1]
        T = tf.convert_to_tensor(T, dtype=tf.float32, name="T")
        # indices of feature corresponding to each datapoint
        ind_K_D = data[2]
        ind_K_D = tf.convert_to_tensor(ind_K_D, dtype=tf.int32, name="ind_K_D")
        ind_T = data[3]
        ind_T = tf.convert_to_tensor(ind_T, dtype=tf.int32, name="ind_T")
        # output to be predicted
        labels = data[7]
        # list of target timestamps to interpolate
        X = data[5]
        X = tf.convert_to_tensor(X, dtype=tf.float32, name="X")
        # counts
        num_distinct_X = data[6]
        num_distinct_X = tf.convert_to_tensor(num_distinct_X, dtype=tf.int32, name="num_distinct_X")
        num_distinct_Y = data[4]
        num_distinct_Y = tf.convert_to_tensor(num_distinct_Y, dtype=tf.int32, name="num_distinct_Y")
        # static data
        static = data[8]
        static = tf.convert_to_tensor(static, dtype=tf.float32, name="static")
        # classes
        classes = data[9]
        # repeat for all MC samples
        classes = np.repeat(classes, self.no_mc_samples)
        IDs = np.repeat(data[10], self.no_mc_samples)
        labels = np.repeat(labels, self.no_mc_samples)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32, name="labels")
        if self.debug:
            return Y, T, ind_K_D, ind_T, num_distinct_Y, X, num_distinct_X, static, labels, classes, IDs
        else:
            return Y, T, ind_K_D, ind_T, num_distinct_Y, X, num_distinct_X, static, labels, classes

    def expand_labels(self, y, alignment=-1, time_window=25):
        y_broad = tf.expand_dims(tf.broadcast_to(tf.expand_dims(y, -1), [y.shape[0], 12 - (alignment + 1)]), -1)
        labelled_time = tf.concat([y_broad, 1 - y_broad], -1)
        early_time = tf.concat([tf.zeros([y.shape[0], time_window - labelled_time.shape[1], 1], dtype=tf.int32),
                                tf.ones([y.shape[0], time_window - labelled_time.shape[1], 1], dtype=tf.int32)], -1)
        return tf.concat([early_time, labelled_time], 1)

    def fast_load(self, features):
        try:
            All = {}
            if features is None:
                date = "19-08-12"
            else:
                date = '19-08-30-{}'.format(features)
            for split in ["train", "val", "test"]:
                path = head + "/data/{}/{}-prep-data-min{}-max{}.pkl".format(split,
                                                                             date,
                                                                             self.min_no_dtpts,
                                                                             self.max_no_dtpts)

                with open(path, "rb") as f:
                    All[split] = pickle.load(f)
            self.train_data = All["train"]
            self.val_data = All["val"]
            self.test_data = All["test"]

        except:
            self.long_load(True, features=features)

    def long_load(self, to_save, features):
        t_print("DataGenerator -- loading data")
        if features is None or features=='rosnati':
            path = self.head + "/data/train/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.train_data = pickle.load(f)
            path = self.head + "/data/val/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.val_data = pickle.load(f)
            path = self.head + "/data/test/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.test_data = pickle.load(f)
        elif features == 'moor':
            path = self.head + "/data/moor/train/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.train_data = pickle.load(f)
            path = self.head + "/data/moor/val/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.val_data = pickle.load(f)
            path = self.head + "/data/moor/test/GP_prep_v2.pkl"
            with open(path, "rb") as f:
                self.test_data = pickle.load(f)

        # shorten TS too long
        self.train_data, no = reduce_data(self.train_data, n_max=self.max_no_dtpts)
        self.val_data, no = reduce_data(self.val_data, n_max=self.max_no_dtpts)
        self.test_data, no = reduce_data(self.test_data, n_max=self.max_no_dtpts)

        # pad data to have same shape
        self.train_data = pad_raw_data(self.train_data)
        self.val_data = pad_raw_data(self.val_data)
        self.test_data = pad_raw_data(self.test_data)

        # augment data to cater for all prediction horizons
        self.train_data = all_horizons(self.train_data)
        self.val_data = all_horizons(self.val_data)
        self.test_data = all_horizons(self.test_data)

        # remove TS too short
        temp = []
        self.train_data, no = reduce_data(self.train_data, n_min=self.min_no_dtpts)
        temp.append(no)
        self.val_data, no = reduce_data(self.val_data, n_min=self.min_no_dtpts)
        temp.append(no)
        self.test_data, no = reduce_data(self.test_data, n_min=self.min_no_dtpts)
        temp.append(no)
        t_print("""Removed patients out of the bound {4} < no_datapoints < {0}.
            Train removed: {1}      Train remaining: {5}
            Val removed:   {2}      Val remaining:   {6}
            Test removed:  {3}      Test remaining:  {7}""".format(self.max_no_dtpts, temp[0], temp[1], temp[2],
                                                                   self.min_no_dtpts,
                                                                   len(self.train_data[4]),
                                                                   len(self.val_data[4]),
                                                                   len(self.test_data[4])))
        del temp

        # extract new indices
        self.train_data = new_indices(self.train_data)
        self.val_data = new_indices(self.val_data)
        self.test_data = new_indices(self.test_data)

        # new data format
        # data = [Y, T, ind_K_D, ind_T, len_T, X, len_X, labels, static, classes, ids, ind_Y]
        # data = [0, 1,       2,     3,     4, 5,     6,      7,      8,       9,  10,    11]
        if to_save:
            All = {"train": self.train_data,
                   "val": self.val_data,
                   "test": self.test_data}
            if features is None:
                date = "19-08-12"
            else:
                date = '19-08-30-{}'.format(features)
            for split in ["train", "val", "test"]:
                path = head + "/data/{}/{}-prep-data-min{}-max{}.pkl".format(split,
                                                                             date,
                                                                             self.min_no_dtpts,
                                                                             self.max_no_dtpts)
                with open(path, "wb") as f:
                    pickle.dump(All[split], f)
