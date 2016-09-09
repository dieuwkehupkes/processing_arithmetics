from keras.callbacks import Callback
import numpy as np


# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    Track different aspects of the network and network performance
    during training.
    """
    def __init__(self, metrics, recurrent_id, param_id=1):
        assert (isinstance(metrics, dict) or isinstance(metrics, list))
        self.recurrent_id = recurrent_id
        self.param_id = param_id
        self.metrics = metrics
        if (isinstance(metrics, dict) and len(metrics) > 1):
            self.mo = True
        else:
            self.mo = False

    def on_train_begin(self, logs={}):

        # if metrics is a dictionary, model has multiple outputs
        if self.mo:
            self.on_train_begin_mo(logs)
        else:
            self.on_train_begin_1o(logs)

        self.esp = []                   # track esp recurrent layer
        self.i = 0                      # track epoch nr

    def on_train_begin_1o(self, logs):

        self.losses = []                # track training loss
        self.val_losses = []            # track validation loss
        self.prediction_error = []      # track training prediction error
        self.metrics_train = dict([(metric, []) for metric in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(metric, []) for metric in self.metrics])         # treck metrics on validation set

    def on_train_begin_mo(self, logs):
        self.losses = dict([(output, []) for output in self.metrics])
        self.val_losses = dict([(output, []) for output in self.metrics])
        self.prediction_error = dict([(output, []) for output in self.metrics])

        self.metrics_train = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set


    def on_epoch_end(self, epoch, logs={}):
        """
        Append tracking objects to their respective lists
        """
        if self.mo:
            self.on_epoch_end_mo(epoch, logs)
        else:
            self.on_epoch_end_1o(epoch, logs)

        self.i += 1

        # compute esp
        recurrent_weights = self.model.layers[self.recurrent_id].get_weights()[self.param_id]
        spec = np.max(np.absolute(np.linalg.eig(recurrent_weights)[0]))
        self.esp.append(spec)

    def on_epoch_end_1o(self, epoch, logs):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        for metric in self.metrics:
            self.metrics_train[metric].append(logs.get(metric))
            self.metrics_val[metric].append(logs.get('val_'+metric))

    def on_epoch_end_mo(self, epoch, logs):
        for output in self.metrics:
            self.losses[output].append(logs.get(output+'_loss'))
            self.losses[output].append(logs.get('val_'+output+'_loss'))

            for metric in self.metrics[output]:
                self.metrics_train[output][metric].append(logs.get(output+'_'+metric))
                self.metrics_val[output][metric].append(logs.get('val_'+output+'_'+metric))



