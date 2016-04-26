from keras.models import Model
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM, Input
from keras.callbacks import EarlyStopping
from TrainingHistory import TrainingHistory
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix

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
        self.model.compile(loss={'output':'mean_squared_error'}, optimizer=self.optimizer)

    def train(self, training_data, batch_size, epochs, validation_data=None, verbosity=1):
        """
        Fit the model.
        """
        history = TrainingHistory()
        X_train, Y_train = training_data

        # fit model
        self.model.fit({'input':X_train}, {'output':Y_train}, validation_data=validation_data, batch_size=batch_size, nb_epoch=epochs, callbacks=[history], verbose=verbosity, shuffle=True)

        self.trainings_history = history            # set trainings_history as attribute


def A2(languages, input_size, size_hidden, size_compare, recurrent, encoding, trainable_embeddings, trainable_comparison, mask_zero, optimizer, validation_split, batch_size, nb_epoch, verbose):
    """
    Write Description.
    """
    # GENERATE TRAINING DATA
    X, Y, N_digits, N_operators = generate_training_data(languages, architecture='A1')

    # SPLIT TRAINING & VALIDATION DATA
    split_at = int(len(X)) * (1. - validation_split)
    X_train, X_val = X[:split_at], X[split_at:]
    Y_train, Y_val = Y[:split_at], Y[split_at:]

    # GENERATE EMBEDDINGS MATRIX
    W_embeddings = generate_embeddings_matrix(N_digits, N_operators, input_size, encoding)
    input_dim = N_operators + N_digits + 2
    input_length = len(X_train[0])

    # CREATE MODEL
    input_layer = Input(shape=(1,), dtype='int32', name='input')        # input layer
    embeddings = Embedding(input_dim=input_dim, output_dim=input_size, input_length=input_length, weights=W_embeddings, mask_zero=mask_zero, trainable=trainable_embeddings, name='embeddings')(input_layer)      # embeddings layer
    recurrent = recurrent(size_hidden, name='recurrent_layer')(embeddings)        # recurrent layer
    comparison = Dense(size_compare, name='comparison')(recurrent)               # comparison layer
    output_layer = Dense(120, activation='softmax', name='output', trainable=trainable_comparison)(comparison)

    model = Model(input=input_layer,  output=output_layer)

    # COMPILE MODEL
    model.compile(loss={'output':'sparse_categorical_crossentropy'}, metrics=['accuracy'], optimizer=optimizer)
    model.summary()

    # TRAIN THE MODEL
    history = TrainingHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit({'input':X_train}, {'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history, early_stopping], validation_data=(X_val, Y_val), shuffle=True)

    return history


def A3(languages, input_size, size_hidden, mask_zero, cotrain, recurrent_layer, size_compare, optimizer, batch_size, nb_epoch, verbose, validation_split, encoding):
    """
    Write Description.
    """
    raise NotImplementedError


def A4(languages, input_size, size_hidden, mask_zero, cotrain, recurrent_layer, size_compare, optimizer, batch_size, nb_epoch, verbose, validation_split, encoding):
    """
    Write Description.
    """
    raise NotImplementedError


