from keras.models import Model
from keras.layers import Embedding, Dense, Input
# from keras.callbacks import EarlyStopping
from TrainingHistory import TrainingHistory
from DrawWeights import DrawWeights
import matplotlib.pyplot as plt

class Training:
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
        self.model.compile(loss={'output': self.loss_function}, optimizer=self.optimizer,
                           metrics=['mean_squared_prediction_error'])

        # print self.model.get_config()

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1):
        """
        Fit the model.
        """
        X_train, Y_train = training_data
        history = TrainingHistory()
        draw_weights = DrawWeights(figsize=(4, 4), layer_id=1, param_id=0)

        # fit model
        self.model.fit({'input':X_train}, {'output':Y_train}, validation_data=validation_data, batch_size=batch_size,
                       nb_epoch=epochs, callbacks=[history, draw_weights], verbose=verbosity, shuffle=True)
        self.loss_function = None

        self.trainings_history = history            # set trainings_history as attribute


class A2(Training):
    """
    Give description.
    """

    def __init__(self, recurrent_layer, input_dim, input_size, input_length, size_hidden, size_compare, W_embeddings,
                 trainable_embeddings=True, trainable_comparison=True, mask_zero=True, optimizer='adagrad'):
        Training.__init__(self, recurrent_layer, input_dim, input_size, input_length, size_hidden, size_compare,
                          W_embeddings, trainable_embeddings=True, trainable_comparison=True, mask_zero=True,
                          optimizer='adagrad')
        self.trainings_history = None            # set trainings_history as attribute

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

        self.trainings_history = history
        

