import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.nn import softmax
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class Tester:
    def __init__(self,
                 model,
                 data,
                 batch_size,
                 log_path,
                 Set
                 ):
        self.data = data
        self.batch_size = batch_size
        self.model = model
        self.log_path = log_path
        self.set = Set
        if Set == 'train':
            self.n_items = len(self.data.train_data[0])
            self.iterator = self.data.next_batch_train_all
        elif Set == 'val' or Set == 'eval':
            self.n_items = len(self.data.val_data[0])
            self.iterator = self.data.next_batch_val_all
        elif Set == 'test':
            self.n_items = len(self.data.test_data[0])
            self.iterator = self.data.next_batch_test_all
        else:
            raise NameError

    def run(self):
        n_batches = int(np.ceil(self.n_items / self.batch_size))
        outcome = {'ID': np.empty(0),
                   'class': np.empty(0),
                   'y': np.empty(0),
                   "y_hat": np.empty(0)}
        for batch in tqdm(range(n_batches)):
            batch_data = next(self.iterator(self.batch_size, batch))
            # expand data
            inputs = batch_data[:8]
            y_hat = softmax(self.model(inputs))
            outcome['ID'] = np.concatenate((outcome['ID'], batch_data[10]))
            outcome['class'] = np.concatenate((outcome['class'], batch_data[9]))
            outcome['y'] = np.concatenate((outcome['y'], batch_data[8].numpy()))
            outcome['y_hat'] = np.concatenate((outcome['y_hat'],
                                               y_hat.numpy()[:, 0]))

        with open(self.log_path, 'wb') as f:
            pickle.dump(outcome, f)


def calc_metrics(outcome):
    df = pd.DataFrame(outcome)
    df['mc_sample'] = df.groupby(['ID', 'class']).cumcount()
    metrics = {}
    n_mc_samples = df['mc_sample'].max()
    for hz in range(6):
        metrics['hz_{}'.format(hz)] = {'AUROC':{'mc_samples':[],},
                                       'PR_AUC':{'mc_samples':[],}}
        for sample in range(n_mc_samples):
            y = df.loc[(df['class'] == hz) & (df.mc_sample == sample), 'y'].to_numpy()
            y_hat = df.loc[(df['class'] == hz) & (df.mc_sample == sample), 'y_hat'].to_numpy()
            # calc metrics
            fpr, tpr, _ = roc_curve(y_true=y, y_score=y_hat)
            roc_auc = auc(fpr, tpr)

            pre, rec, _ = precision_recall_curve(y_true=y, probas_pred=y_hat)
            recall = rec[np.argsort(rec)]
            precision = pre[np.argsort(rec)]
            pr_auc = auc(recall, precision)

            metrics['hz_{}'.format(hz)]['AUROC']['mc_samples'].append(roc_auc)
            metrics['hz_{}'.format(hz)]['PR_AURC']['mc_samples'].append(pr_auc)
        for m in ['AUROC', 'PR_AUC']:
            metrics['hz_{}'.format(hz)][m]['mean'] = np.mean(metrics['hz_{}'.format(hz)][m]['mc_samples'])
            metrics['hz_{}'.format(hz)][m]['std'] = np.std(metrics['hz_{}'.format(hz)][m]['mc_samples'])
    return metrics
