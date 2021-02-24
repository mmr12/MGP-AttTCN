import os
import sys
import numpy as np
from scipy import interp
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print


def evals(y_true, y_proba, classes, cv=False, singles=True, overall=False):
    roc_auc = []
    pr_auc = []
    roc_comps = []
    pr_comps = []
    if singles:
        # calculate ROC and PR for each horizon
        cl_print = []
        for i in range(7):
            idx = classes == i
            if np.sum(idx) != 0:
                cl_print.append(np.sum(idx))
                y_star = y_true[idx]
                y = y_proba[idx, 0]
                roc_horizon, pr_horizon, roc_comp, pr_comp = one_eval(y_star, y, cv=cv)
                if cv:
                    roc_comps.append(np.asarray(roc_comp))
                    pr_comps.append(np.asarray(pr_comp))
                roc_auc.append(roc_horizon)
                pr_auc.append(pr_horizon)
            else:
                cl_print.append(0)
                #t_print("warning: no class {}".format(i))
                roc_auc.append(0)
                pr_auc.append(0)
        #print('classes', cl_print, np.sum(cl_print), flush=True)
    if overall:
        # calculate ROC and PR over all horizons
        roc_horizon, pr_horizon, roc_comp, pr_comp = one_eval(y_star=y_true, y=y_proba[:, 0], cv=cv)
        if cv:
            roc_comps.append(np.asarray(roc_comp))
            pr_comps.append(np.asarray(pr_comp))
        roc_auc.append(roc_horizon)
        pr_auc.append(pr_horizon)
    return roc_auc, pr_auc, roc_comps, pr_comps


def one_eval(y_star, y, cv=False):
    linear_space = np.linspace(0, 1, 100)

    fpr, tpr, _ = roc_curve(y_true=y_star, y_score=y)
    roc_auc = auc(fpr, tpr)

    pre, rec, _ = precision_recall_curve(y_true=y_star, probas_pred=y)
    recall = rec[np.argsort(rec)]
    precision = pre[np.argsort(rec)]
    pr_auc = auc(recall, precision)

    adj_fpr, adj_tpr = linear_space, interp(linear_space, fpr, tpr)
    adj_tpr[0] = 0.0
    adj_rec, adj_pre = linear_space, interp(linear_space, recall, precision)
    if cv:
        return roc_auc, pr_auc, [adj_fpr, adj_tpr], [adj_rec, adj_pre]
    else:
        return roc_auc, pr_auc, None, None
