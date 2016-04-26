from keras.models import Model
# from keras.metrics import binary_accuracy, mean_squared_error
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM, Input
from keras.callbacks import EarlyStopping
from TrainingHistory import TrainingHistory
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix
import matplotlib.pyplot as plt
import theano

class Training():
    """
    Give elaborate description
    """
    def __init__(self, recurrent_layer, input_dim, input_size, input_length, size_hidden, size_compare, W_embeddings, trainable_embeddings=True, trainable_comparison=True, mask_zero=True, optimizer='adagrad'):

        # set attributes
        self.recurrent_layer = recurrent_layer
        self.input_dim = input_dim
        self.input_size = input_size
        self.input_length = input_length
        self.size_hidden = size_hidden
        self.size_compare = size_compare
        self.cotrain_comparison = trainable_comparison
        self.cotrain_embeddings = trainable_embeddings
        self.mask_zero = mask_zero
        self.optimizer = optimizer
        
        # build model
        self._build(W_embeddings)

    def _build(self, W_embeddings):
        pass

    def model_summary(self):
        print(self.model.summary())

    def visualise_embeddings(self):
        raise NotImplementedError()

    def plot_training_history(self, save_to_file=False):
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
        plt.show()

        if save_to_file:
            raise NotImplementedError()


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
        embeddings = Embedding(input_dim=self.input_dim, output_dim=self.input_size, input_length=self.input_length, weights=W_embeddings, mask_zero=self.mask_zero, trainable=self.cotrain_embeddings, name='embeddings')(input_layer) 
        
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
        self.model.compile(loss={'output':self.loss_function}, optimizer=self.optimizer, metrics=['accuracy'])

        print self.model.get_config()

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1):
        """
        Fit the model.
        """
        X_train, Y_train = training_data
        history = TrainingHistory()

        # fit model
        self.model.fit({'input':X_train}, {'output':Y_train}, validation_data=validation_data, batch_size=batch_size, nb_epoch=epochs, callbacks=[history], verbose=verbosity, shuffle=True)

        self.trainings_history = history            # set trainings_history as attribute

    def discrete_prediction(y_true, y_pred):
        return theano.tensor.mean(theano.tensor.square(y_pred-y_true), axis=1)


class A2(Training):
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
        embeddings = Embedding(input_dim=self.input_dim, output_dim=self.input_size, input_length=self.input_length, weights=W_embeddings, mask_zero=self.mask_zero, trainable=self.cotrain_embeddings, name='embeddings')(input_layer) 
        
        # create recurrent layer
        recurrent = self.recurrent_layer(self.size_hidden, name='recurrent_layer')(embeddings)

        # create comparison layer
        comparison = Dense(self.size_compare, name='comparison', trainable=self.cotrain_comparison)(recurrent)

        # create output layer
        output_layer = Dense(120, activation='softmax', name='output')(comparison)

        # create model
        self.model = Model(input=input_layer, output=output_layer)

        # compile
        self.loss_function = 'sparse_categorical_crossentropy'
        self.model.compile(loss={'output':self.loss_function}, optimizer=self.optimizer)

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1):
        """
        Fit the model.
        """
        history = TrainingHistory()
        X_train, Y_train = training_data

        # fit model
        self.model.fit({'input':X_train}, {'output':Y_train}, validation_data=validation_data, batch_size=batch_size, nb_epoch=epochs, callbacks=[history], verbose=verbosity, shuffle=True)

        self.trainings_history = history            # set trainings_history as attribute
        

