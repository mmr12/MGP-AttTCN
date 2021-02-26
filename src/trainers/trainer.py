import os
import pickle
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.loss_n_eval.aucs import evals as uni_evals
from src.loss_n_eval.losses import grad, GP_loss
from src.utils.debug import t_print

class Trainer:
    def __init__(self,
                 model,
                 data,
                 num_epochs,
                 batch_size,
                 optimizer,
                 global_step,
                 summary_writers,
                 log_path,
                 train_only=False,
                 notebook_friendly=False,
                 eval_every=20,
                 late_patients_only=False,
                 horizon0=False,
                 lab_vitals_only=False,
                 weighted_loss=None,
                 ):

        self.model = model
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.global_step = global_step
        self.notebook_friendly = notebook_friendly
        self.summary_writers = summary_writers
        self.log_path = log_path
        self.train_only = train_only
        self.eval_every = eval_every
        self.late_patients_only = late_patients_only
        self.horizon0 = horizon0
        self.lab_vitals_only = lab_vitals_only
        self.weighted_loss = weighted_loss

        # Initialise progress trackers - epoch
        self.train_loss_results = []
        self._roc = []
        self._pr = []

        # Initialise progress trackers - batch
        self.train_loss_results_batch = []
        self._roc_batch = []
        self._pr_batch = []

        if self.late_patients_only:
            # 'int' truncates, hence int + 1 finds the ceiling
            self.no_batches = int(len(self.data.late_case_patients) * 6 / self.batch_size) + 1
        else:
            self.no_batches = int(len(self.data.train_case_idx) / self.batch_size) + 1
        self.no_dev_batches = int(len(self.data.val_data[-1]) / self.batch_size) + 1


    def run(self):
        for epoch in range(self.num_epochs):
            t_print("Start of epoch {}".format(epoch))
            # shuffle data
            np.random.shuffle(self.data.train_case_idx)
            np.random.shuffle(self.data.train_control_idx)
            self.data.apply_reshuffle()

            for batch in tqdm(range(self.no_batches)):
                # Load data
                # batch_data = Y, T, ind_features, num_distinct_Y, X, num_distinct_X, static, labels, classes
                batch_data = next(self.data.next_batch(self.batch_size, batch, late=self.late_patients_only,
                                                       horizon0=self.horizon0))
                # batch_data[8] is static
                if self.lab_vitals_only:
                    inputs = batch_data[:7]
                else:
                    inputs = batch_data[:8]
                y = batch_data[8]
                classes = batch_data[9]
                if len(y) > 0:

                    # Evaluate loss and gradient
                    loss_value, grads = grad(self.model, inputs, y, GP=True, weighted_loss=self.weighted_loss)
                    # Apply gradient
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.global_step)
                    self.global_step.assign_add(1)

                    # Track progress - loss
                    self.train_loss_results_batch.append(loss_value.numpy())

                    # Track progress - metrics
                    y_hat = tf.nn.softmax(self.model(inputs))
                    roc_auc, pr_auc, _, _ = uni_evals(y.numpy(), y_hat.numpy(), classes, overall=True)

                    # write into tensorboard
                    step = (epoch * self.no_batches + batch) * self.no_dev_batches
                    with self.summary_writers['train'].as_default():
                        tf.summary.scalar('loss', loss_value.numpy(), step=step)
                        for i in range(8):
                            tf.summary.scalar("roc_{}".format(i), roc_auc[i], step=step)
                            tf.summary.scalar("pr_{}".format(i), pr_auc[i], step=step)

                    if batch % self.eval_every == 0:
                        t_print("Epoch {:03d} -- Batch {:03d}: Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(
                            epoch, batch, loss_value.numpy(), roc_auc[7], pr_auc[7]))
                        if not self.train_only:
                            # iterate over all horizons
                            for horizon in range(7):
                                self.dev_eval_per_horizon(horizon, step)

            # end of batch loop
            self.train_loss_results.append(np.mean(self.train_loss_results_batch))
            self._roc.append(np.mean(np.asarray(self._roc_batch), axis=0))
            self._pr.append(np.mean(np.asarray(self._pr_batch), axis=0))
            t_print("End of epoch {:03d}: Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(
                epoch, self.train_loss_results[-1], self._roc[-1][7], self._pr[-1][7]))

            if not self.train_only:
                # save all outputs
                all_dev_y = []
                all_dev_y_hat = []
                classes = []
                for dev_batch in range(self.no_dev_batches):
                    step = (self.num_epochs * self.no_batches) * self.no_dev_batches
                    y_true, y_hat, _class = self.dev_eval(dev_batch, step)
                    all_dev_y.append(y_true)
                    all_dev_y_hat.append(y_hat)
                    classes.append(_class)
                if not self.notebook_friendly:
                    _to_save = {"epoch": epoch,
                                "y_true": all_dev_y,
                                "y_hat": all_dev_y_hat,
                                "classes": classes,
                                "weights": self.model.get_weights()}
                    with open(os.path.join(self.log_path, 'epoch_{}_out.pkl'.format(epoch)), "wb") as f:
                        pickle.dump(_to_save, f)

    def dev_eval_per_horizon(self, horizon, step):
        batch_data = next(self.data.next_batch_dev_small(horizon))
        _, loss_dev, roc_auc, pr_auc = self.step(batch_data)
        with self.summary_writers['val'].as_default():
            tf.summary.scalar("loss_dev", loss_dev.numpy(), step=step + horizon)
            tf.summary.scalar("roc_{}_dev".format(horizon), roc_auc[horizon], step=step + horizon)
            tf.summary.scalar("pr_{}_dev".format(horizon), pr_auc[horizon], step=step + horizon)
        # print
        t_print("DEV hz {} Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(horizon,
                                                                                loss_dev, roc_auc[7], pr_auc[7]))

    def dev_eval(self, dev_batch, step):
        batch_data = next(self.data.next_batch_dev_all(self.batch_size, dev_batch))
        dev_y_hat, loss_dev, roc_auc, pr_auc = self.step(batch_data)
        # write into sacred observer
        with self.summary_writers['val'].as_default():
            tf.summary.scalar("loss_dev", loss_dev.numpy(), step=step + dev_batch)
            for i in range(7):
                if roc_auc[i] != 0: tf.summary.scalar("roc_{}_dev".format(i), roc_auc[i], step=step + dev_batch)
                if pr_auc[i] != 0: tf.summary.scalar("pr_{}_dev".format(i), pr_auc[i], step=step + dev_batch)
        # print
        t_print("DEV Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(loss_dev, roc_auc[7], pr_auc[7]))
        # return y_true, y_hat, class
        return np.array(batch_data[9]), dev_y_hat, np.array(batch_data[9])

    def step(self, batch_data):
        # batch_data[8] is static
        if self.lab_vitals_only:
            inputs = batch_data[:7]
        else:
            inputs = batch_data[:8]
        y = batch_data[8]
        classes = batch_data[9]
        if len(y) > 0:
            # Track progress - dev loss
            loss_dev = GP_loss(self.model, inputs, y)
            # Track progress - dev metrics
            dev_y_hat = tf.nn.softmax(self.model(inputs))
            roc_auc, pr_auc, _, _ = uni_evals(y.numpy(), dev_y_hat.numpy(), classes, overall=True)
            return dev_y_hat, loss_dev, roc_auc, pr_auc
        else: return None, None, None, None

