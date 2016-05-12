from keras.models import Model
from keras.layers import Embedding, Dense, Input
# from keras.callbacks import EarlyStopping
from TrainingHistory import TrainingHistory
from DrawWeights import DrawWeights
from PlotEmbeddings import PlotEmbeddings
from MonitorUpdates import MonitorUpdates
from Logger import Logger
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class Training(object):
    """
    Give elaborate description
    """
    def __init__(self, recurrent_layer, input_dim, input_size, input_length, size_hidden, size_compare,
                 W_embeddings, dmap, trainable_embeddings=True, trainable_comparison=True, mask_zero=True,
                optimizer='adagrad'):

        # set attributes
        self.recurrent_layer = recurrent_layer
        self.input_dim = input_dim
        self.input_size = input_size
        self.input_length = input_length
        self.size_hidden = size_hidden
        self.size_compare = size_compare
        self.dmap = dmap
        self.cotrain_comparison = trainable_comparison
        self.cotrain_embeddings = trainable_embeddings
        self.mask_zero = mask_zero
        self.optimizer = optimizer
        self.trainings_history = None
        self.model = None
        self.loss_function = None

        # build model
        self._build(W_embeddings)

    def _build(self, W_embeddings):
        pass

    def model_summary(self):
        print(self.model.summary())

    def visualise_embeddings(self):
        raise NotImplementedError()

    def save_to_file(self, filename):
        """Save model to file"""
        json_string = self.model.to_json()
        f = open(filename, 'w')
        f.write(json_string)
        self.model.save(filename+'_weights.h5')
        f.close()

    def plot_prediction_error(self, save_to_file=False):
        """
        Plot the prediction error during the last training
        round of the network
        :param save_to_file:    file name to save file to
        """
        plt.plot(self.trainings_history.prediction_error, label="Training set")
        plt.plot(self.trainings_history.val_prediction_error, label="Validation set")
        plt.title("Prediction error during last training round")
        plt.xlabel("Epoch")
        plt.ylabel("Prediction Error")
        plt.axhline(xmin=0)
        plt.legend()
        plt.show()

    def plot_loss(self, save_to_file=False):
        """
        Plot loss on the last training
        of the network.
        """
        plt.plot(self.trainings_history.losses, label='Training set')
        plt.plot(self.trainings_history.val_losses, label='Validation set')
        plt.title("Loss during last training")
        plt.xlabel("Epoch")
        plt.ylabel(self.loss_function)
        plt.axhline(xmin=0)
        plt.legend()
        plt.show()

    def plot_embeddings(self):
        """
        Plot embeddings of the network (only available for
        2 dimensional embeddings)
        :return:
        """
        weights = self.model.layers[1].get_weights()[0]
        assert weights.shape[1] == 2, "visualise embeddings only available for 2d embeddings"
        # find limits
        xmin, ymin = 1.1 * weights.max(axis=0)
        xmax, ymax = 1.1 * weights.min(axis=0)
        # use dmap to determine labels
        dmap_inverted = dict(zip(self.dmap.values(), self.dmap.keys()))
        i = 0
        for weight_set in weights:
            xy = tuple(weight_set)
            print xy
            x, y = xy
            plt.plot(x, y, 'o')
            plt.annotate(dmap_inverted[i], xy=xy)
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            i+=1
        plt.show()

    def generate_callbacks(self, embeddings_animation, plot_embeddings, print_every):
        """
        Generate sequence of callbacks to use during training
        :param embeddings_animation:        set to true to generate visualisation of embeddings
        :param plot_embeddings:             generate scatter plot of embeddings every plot_embeddings epochs
        :param print_every:                 print summary of results every print_every epochs
        :return:
        """

        history = TrainingHistory()
        callbacks = [history]

        if embeddings_animation:
            draw_weights = DrawWeights(figsize=(4, 4), layer_id=1, param_id=0)
            callbacks.append(draw_weights)

        if plot_embeddings:
            if plot_embeddings == True:
                pass
            else:
                embeddings_plot = PlotEmbeddings(plot_embeddings, self.dmap)
                callbacks.append(embeddings_plot)

        if print_every:
            logger = Logger(print_every)
            callbacks.append(logger)

        return callbacks


class A1(Training):
    """
    Give description.
    """

    def _build(self, W_embeddings):
        """
        Build the trainings architecture around
        the model.
        """
        # create input layer
        input_layer = Input(shape=(1,), dtype='int32', name='input')

        # create embeddings
        embeddings = Embedding(input_dim=self.input_dim, output_dim=self.input_size,
                               input_length=self.input_length, weights=W_embeddings,
                               mask_zero=self.mask_zero, trainable=self.cotrain_embeddings,
                               name='embeddings')(input_layer)
        
        # create recurrent layer
        recurrent = self.recurrent_layer(self.size_hidden, name='recurrent_layer')(embeddings)

        # create comparison layer
        comparison = Dense(self.size_compare, name='comparison', trainable=self.cotrain_comparison)(recurrent)

        # create output layer
        output_layer = Dense(1, activation='linear', name='output')(comparison)

        # create model
        self.model = Model(input=input_layer, output=output_layer)

        # compile
        self.loss_function = 'mean_squared_error'
        self.model.compile(loss={'output': self.loss_function}, optimizer=self.optimizer,
                           metrics=['mspe'])

        # print self.model.get_config()

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1,
              embeddings_animation=False, plot_embeddings=False, logger=False):
        """
        Fit the model.
        :param embeddings_animation:    Set to true to create an animation of the development of the embeddings
                                        after training.
        :param plot_embeddings:        Set to N to plot the embeddings every N epochs, only available for 2D
                                        embeddings.
        """
        X_train, Y_train = training_data

        callbacks = self.generate_callbacks(embeddings_animation, plot_embeddings, logger)

        # fit model
        self.model.fit({'input': X_train}, {'output': Y_train}, validation_data=validation_data, batch_size=batch_size,
                       nb_epoch=epochs, callbacks=callbacks, verbose=verbosity, shuffle=True)
        self.loss_function = None

        self.trainings_history = callbacks[0]            # set trainings_history as attribute

