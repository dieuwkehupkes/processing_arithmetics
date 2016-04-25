from keras.models import Model
from keras.layers import SimpleRNN, Embedding, Dense, GRU, LSTM, Input
from keras.callbacks import EarlyStopping
from TrainingHistory import TrainingHistory
from generate_training_data import generate_training_data
from auxiliary_functions import generate_embeddings_matrix

def A1(languages, input_size, size_hidden, size_compare, recurrent, encoding, trainable_embeddings, trainable_comparison, mask_zero, optimizer, validation_split, batch_size, nb_epoch, verbose):

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
    comparison = Dense(size_compare, name='comparison', trainable=trainable_comparison)(recurrent)               # comparison layer
    output_layer = Dense(1, activation='linear', name='output')(comparison)

    model = Model(input=input_layer,  output=output_layer)

    # COMPILE MODEL
    model.compile(loss={'output':'mean_squared_error'}, metrics=['accuracy'], optimizer=optimizer)
    model.summary()

    # print("predictions and targets before training:\n")
    # predictions = model.predict(X_val)
    # for prediction in xrange(len(predictions)):
        # print('%s\t\t%s' % (str(predictions[prediction][0]), str(Y_val[prediction])))

    # TRAIN THE MODEL
    history = TrainingHistory()
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit({'input':X_train}, {'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history], validation_data=(X_val, Y_val), shuffle=True)
    # model.fit({'input':X_train}, {'output':Y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[history], validation_split=0.1, shuffle=True)

    print("\n\n\npredictions after training:\n")
    predictions = model.predict(X_val)
    for prediction in xrange(len(predictions)):
        print('%s\t\t%s' % (str(predictions[prediction][0]), str(Y_val[prediction])))

    return history


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


