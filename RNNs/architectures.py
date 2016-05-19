from keras.models import Model, model_from_json
from keras.layers import Embedding, Dense, Input, merge
import keras.preprocessing.sequence
from generate_training_data import generate_treebank
from TrainingHistory import TrainingHistory
from DrawWeights import DrawWeights
from PlotEmbeddings import PlotEmbeddings
from Logger import Logger
import matplotlib.pyplot as plt
import numpy as np
import random


class Training(object):
    """
    Give elaborate description
    """
    def __init(self):
        """
        Create training architecture
        """

    def generate_model(self, recurrent_layer, input_dim, input_size, input_length, size_hidden, size_compare,
                 W_embeddings, dmap, trainable_embeddings=True, trainable_comparison=True, mask_zero=True,
                 dropout_recurrent=0.0, optimizer='adagrad'):
        """
        Generate the model to be trained
        :param recurrent_layer:     type of recurrent layer (from keras.layers SimpleRNN, GRU or LSTM)
        :param input_dim:           vocabulary size
        :param input_size:          dimensionality of the embeddings (input size recurrent layer)
        :param input_length:        max sequence length
        :param size_hidden:         size recurrent layer
        :param size_compare:        size comparison layer
        :param W_embeddings:        Either an embeddings matrix or None if to be generate by keras layer
        :param dmap:                A map from vocabulary words to integers
        :param trainable_embeddings: set to false to fix embedding weights during training
        :param trainable_comparison: set to false to fix comparison layer weights during training
        :param mask_zero:            set to true to mask 0 values
        :param dropout_recurrent:    dropout param for recurrent weights
        :param optimizer:            optimizer to use during training
        :return:
        """

        # set network attributes
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
        self.dropout_recurrent = dropout_recurrent
        self.optimizer = optimizer
        self.trainings_history = None
        self.model = None

        # build model
        self._build(W_embeddings)

    def _build(self, W_embeddings):
        raise NotImplementedError()

    def add_pretrained_model(self, json_model, model_weights, optimizer, dmap):
        """
        Add a model with already trained weights
        :param json_model:      json filename containing model architecture
        :param model_weights:   h5 file containing model weights
        :param optimizer:       optimizer to use during training
        """
        self.model = model_from_json(open(json_model).read())
        self.model.load_weights(model_weights)
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=self.metrics)
        self.dmap = dmap

        return

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

    def plot_esp(self):
        """
        Plot the spectral radius of the recurrent connections
        of the network during training
        """
        plt.plot(self.trainings_history.esp)
        plt.title("Spectral radius of recurrent connections")
        plt.xlabel("Epoch")
        plt.ylabel("spectral radius")
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
            i += 1
        plt.show()

    def generate_callbacks(self, weights_animation, plot_embeddings, print_every):
        """
        Generate sequence of callbacks to use during training
        :param weights_animation:        set to true to generate visualisation of embeddings
        :param plot_embeddings:             generate scatter plot of embeddings every plot_embeddings epochs
        :param print_every:                 print summary of results every print_every epochs
        :return:
        """

        history = TrainingHistory()
        callbacks = [history]

        if weights_animation:
            layer_id, param_id = weights_animation
            draw_weights = DrawWeights(figsize=(4, 4), layer_id=layer_id, param_id=param_id)
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
    def __init__(self):
        self.loss_function = 'mean_squared_error'
        self.metrics = ['mspe']

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
        recurrent = self.recurrent_layer(self.size_hidden, name='recurrent_layer',
                                         dropout_U=self.dropout_recurrent)(embeddings)

        # create comparison layer
        comparison = Dense(self.size_compare, name='comparison', trainable=self.cotrain_comparison)(recurrent)

        # create output layer
        output_layer = Dense(1, activation='linear', name='output')(comparison)

        # create model
        self.model = Model(input=input_layer, output=output_layer)

        # compile
        self.model.compile(loss={'output': self.loss_function}, optimizer=self.optimizer,
                           metrics=self.metrics)

        # print self.model.get_config()

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1,
              weights_animation=False, plot_embeddings=False, logger=False):
        """
        Fit the model.
        :param embeddings_animation:    Set to true to create an animation of the development of the embeddings
                                        after training.
        :param plot_embeddings:        Set to N to plot the embeddings every N epochs, only available for 2D
                                        embeddings.
        """
        X_train, Y_train = training_data

        callbacks = self.generate_callbacks(weights_animation, plot_embeddings, logger)

        # fit model
        self.model.fit({'input': X_train}, {'output': Y_train}, validation_data=validation_data,
                       batch_size=batch_size, nb_epoch=epochs, callbacks=callbacks,
                       verbose=verbosity, shuffle=True)

        self.trainings_history = callbacks[0]            # set trainings_history as attribute

    def generate_training_data(self, languages, dmap, digits, pad_to=None):
        """
        Take a dictionary that maps languages to number of sentences and
         return numpy arrays with training data.
        :param languages:       dictionary mapping languages (str name) to numbers
        :param architecture:    architecture for which to generate training data
        :param pad_to:          length to pad training data to
        :return:                tuple, input, output, number of digits, number of operators
                                map from input symbols to integers
        """
        # generate treebank with examples
        treebank = generate_treebank(languages)
        random.shuffle(treebank.examples)

        # create empty input and targets
        X, Y = [], []

        # loop over examples
        for expression, answer in treebank.examples:
            input_seq = [dmap[i] for i in str(expression).split()]
            answer = str(answer)
            X.append(input_seq)
            Y.append(answer)

        # pad sequences to have the same length
        assert pad_to == None or len(X[0]) <= pad_to, 'length test is %i, max length is %i. Test sequences should not be truncated' % (len(X[0]), pad_to)
        X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='int32', maxlen=pad_to)

        return X_padded, np.array(Y)


class A4(Training):
    """
    Class where comparison of two different arithmetic
    expressions is used as a training signal
    """
    def __init__(self):
        self.loss_function = 'categorical_crossentropy'
        self.metrics = ['categorical_accuracy']

    def _build(self, W_embeddings):
        """
        Build the trainings architecture around
        the model.
        """
        # create input layer
        input1 = Input(shape=(1,), dtype='int32', name='input1')
        input2 = Input(shape=(1,), dtype='int32', name='input2')

        # create embeddings
        embeddings_layer = Embedding(input_dim=self.input_dim, output_dim=self.input_size,
                                     input_length=self.input_length, weights=W_embeddings,
                                     mask_zero=self.mask_zero, trainable=self.cotrain_embeddings,
                                     name='embeddings')

        # create recurrent layer
        recurrent_layer = self.recurrent_layer(self.size_hidden, name='recurrent_layer',
                                               dropout_U=self.dropout_recurrent)

        # create embeddings for both inputs
        embeddings1 = embeddings_layer(input1)
        embeddings2 = embeddings_layer(input2)

        # create recurrent rperesentations for both inputs
        recurrent1 = recurrent_layer(embeddings1)
        recurrent2 = recurrent_layer(embeddings2)

        # concatenate output
        concat = merge([recurrent1, recurrent2], mode='concat', concat_axis=-1)

        # create output layer
        output_layer = Dense(3, activation='softmax', name='output')(concat)

        # create model
        self.model = Model(input=[input1, input2], output=output_layer)

        # compile
        self.model.compile(loss={'output': self.loss_function}, optimizer=self.optimizer,
                           metrics=self.metrics)

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1,
              weights_animation=False, plot_embeddings=False, logger=False):
        """
        Fit the model.
        :param embeddings_animation:    Set to true to create an animation of the development of the embeddings
                                        after training.
        :param plot_embeddings:        Set to N to plot the embeddings every N epochs, only available for 2D
                                        embeddings.
        """
        X_train, Y_train = training_data
        X1_train, X2_train = X_train

        callbacks = self.generate_callbacks(weights_animation, plot_embeddings, logger)

        # fit model
        self.model.fit({'input1': X1_train, 'input2': X2_train}, {'output': Y_train},
                       validation_data=validation_data, batch_size=batch_size, nb_epochs=epochs,
                       callbacks=callbacks, verbose=verbosity, shuffle=True)

        self.trainings_history = callbacks[0]

    def generate_training_data(self, languages, dmap, digits, pad_to=None):
        """
        Take a dictionary that maps languages to number of sentences and
         return numpy arrays with training data.
        :param languages:       dictionary mapping languages (str name) to numbers
        :param architecture:    architecture for which to generate training data
        :param pad_to:          length to pad training data to
        :return:                tuple, input, output, number of digits, number of operators
                                map from input symbols to integers
        """
        # generate treebank with examples
        treebank1 = generate_treebank(languages)
        random.shuffle(treebank1.examples)
        treebank2 = generate_treebank(languages)
        random.shuffle(treebank2.examples)

        # create empty input and targets
        X1, X2, Y = [], [], []

        # loop over examples
        for example1, example2 in zip(treebank1.examples, treebank2.examples):
            expr1, answ1 = example1
            expr2, answ2 = example2
            input_seq1 = [dmap[i] for i in str(expr1).split()]
            input_seq2 = [dmap[i] for i in str(expr2).split()]
            answer = np.argmax([answ1 < answ2, answ1 == answ2, answ1 > answ2])
            X1.append((input_seq1))
            X2.append((input_seq2))
            Y.append(answer)

        # pad sequences to have the same length
        assert pad_to == None or len(X1[0]) <= pad_to, \
            'length test is %i, max length is %i. Test sequences should not be truncated' \
            % (len(X1[0]), pad_to)
        X1_padded = keras.preprocessing.sequence.pad_sequences(X1, dtype='int32', maxlen=pad_to)
        X2_padded = keras.preprocessing.sequence.pad_sequences(X2, dtype='int32', maxlen=pad_to)

        X_padded = zip(X1_padded, X2_padded)

        return X_padded, np.array(Y)
