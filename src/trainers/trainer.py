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
                 _run,
                 train_only=False,
                 notebook_friendly=False,
                 log_path=head + "'experiments/19-08-09-GP_CNN/tf_log",
                 eval_every=20,
                 late_patients_only=False,
                 horizon0=False,
                 lab_vitals_only=False,
                 weighted_loss=None):

        self.model = model
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.global_step = global_step
        self.notebook_friendly = notebook_friendly
        if not self.notebook_friendly:
            self._run = _run
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

        self.dev_loss_results = []
        self._roc_dev = []
        self._pr_dev = []
        self.dev_step = []

        self.dev_loss_results_batch = []
        self._roc_dev_batch = []
        self._pr_dev_batch = []
        self.dev_step_batch = []

        self.best_epoch = 0
        self.best_pr = 0

        self.check_roc = 0
        self.check_epoch = 0

        if self.late_patients_only:
            # 'int' truncates, hence int + 1 finds the ceiling
            self.no_batches = int(len(self.data.late_case_patients) * 6 / self.batch_size) + 1
        else:
            self.no_batches = int(len(self.data.train_case_idx) / self.batch_size) + 1
        self.no_dev_batches = int(len(self.data.val_data[-1]) / self.batch_size) + 1

        # final outputs
        self.train_results = {
            "loss": np.asarray(self.train_loss_results),
            "au_roc": np.asarray(self._roc),
            "au_pr": np.asarray(self._pr),
            #
            "loss_batch": np.asarray(self.train_loss_results_batch),
            "au_roc_batch": np.asarray(self._roc_batch),
            "au_pr_batch": np.asarray(self._pr_batch)
        }

        self.dev_results = {
            "loss": np.asarray(self.dev_loss_results),
            "au_roc": np.asarray(self._roc_dev),
            "au_pr": np.asarray(self._pr_dev),
            "epochs": np.asarray(self.dev_step),
            #
            "loss_batch": np.asarray(self.dev_loss_results_batch),
            "au_roc_batch": np.asarray(self._roc_dev_batch),
            "au_pr_batch": np.asarray(self._pr_dev_batch)

        }

        self.best_results = {
            "avg_pr": self.best_pr,
            "epoch": self.best_epoch
        }


    def run(self):
        for epoch in range(self.num_epochs):
            t_print("Start of epoch {}".format(epoch))
            # shuffle data
            np.random.shuffle(self.data.train_case_idx)
            np.random.shuffle(self.data.train_control_idx)
            self.data.apply_reshuffle()

            for batch in tqdm(range(self.no_batches)):
                if batch % 5 == 0:
                    t_print("Start of batch {}".format(batch))
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
                    self._roc_batch.append(roc_auc)
                    self._pr_batch.append(pr_auc)

                    # write into sacred observer
                    step = (epoch * self.no_batches + batch) * self.no_dev_batches
                    if not self.notebook_friendly:
                        self._run.log_scalar("loss", loss_value.numpy(), step=step)
                        for i in range(8):
                            self._run.log_scalar("roc_{}".format(i), roc_auc[i], step=step)
                            self._run.log_scalar("pr_{}".format(i), pr_auc[i], step=step)

                    if batch % self.eval_every == 0:
                        t_print("Epoch {:03d} -- Batch {:03d}: Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(
                            epoch, batch, loss_value.numpy(), roc_auc[7], pr_auc[7]))
                        if not self.train_only:
                            # iterate over all horizons
                            for dev_batch in range(7):
                                self.dev_eval(epoch, batch, dev_batch, step)

            # end of batch loop
            self.train_loss_results.append(np.mean(self.train_loss_results_batch))
            self._roc.append(np.mean(np.asarray(self._roc_batch), axis=0))
            self._pr.append(np.mean(np.asarray(self._pr_batch), axis=0))
            t_print("End of epoch {:03d}: Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(
                epoch, self.train_loss_results[-1], self._roc[-1][7], self._pr[-1][7]))

            if not self.train_only:
                # save all outputs
                self.all_dev_y = []
                self.all_dev_y_hat = []
                for dev_batch in range(self.no_dev_batches):
                    step = (self.num_epochs * self.no_batches) * self.no_dev_batches
                    self.dev_eval(self.num_epochs, None, dev_batch, step)
                if not self.notebook_friendly:
                    _to_save = {"epoch": epoch,
                                "y_hat": self.all_dev_y_hat,
                                "weights": self.model.get_weights()}
                    with open(head + "/save_temp.pkl", "wb") as f:
                        pickle.dump(_to_save, f)
                    self._run.add_artifact(head + "/save_temp.pkl", "epoch_{}_dict.pkl".format(epoch))

    def dev_eval(self, epoch, batch, dev_batch, step):
        if batch is not None:
            batch_data = next(self.data.next_batch_dev_small(dev_batch))
        else:
            batch_data = next(self.data.next_batch_dev_all(self.batch_size, dev_batch))
            self.all_dev_y.append(batch_data[8].numpy())
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
            if batch is None:
                self.dev_loss_results.append([dev_batch, loss_dev.numpy()])
            else:
                self.dev_loss_results_batch.append([dev_batch, loss_dev.numpy()])

            # Track progress - dev metrics
            dev_y_hat = tf.nn.softmax(self.model(inputs))
            roc_auc, pr_auc, _, _ = uni_evals(y.numpy(), dev_y_hat.numpy(), classes, overall=True)
            if batch is None:
                self.all_dev_y_hat.append(dev_y_hat.numpy())
                self._roc_dev.append([dev_batch] + roc_auc)
                self._pr_dev.append([dev_batch] + pr_auc)
            else:
                self._roc_dev_batch.append([dev_batch] + roc_auc)
                self._pr_dev_batch.append([dev_batch] + pr_auc)

                # Iteration storage
                self.dev_step_batch.append([dev_batch, epoch, batch])

                # write into sacred observer
                if not self.notebook_friendly:
                    self._run.log_scalar("loss_dev", loss_dev.numpy(), step=step + dev_batch)
                    for i in range(7):
                        self._run.log_scalar("roc_{}_dev".format(i), roc_auc[i], step=step + dev_batch)
                        self._run.log_scalar("pr_{}_dev".format(i), pr_auc[i], step=step + dev_batch)

            # print
            t_print("DEV Loss: {:.3f}\tROC o/a:{:.3f}\tPR  o/a:{:.3f}".format(loss_dev, roc_auc[7], pr_auc[7]))
