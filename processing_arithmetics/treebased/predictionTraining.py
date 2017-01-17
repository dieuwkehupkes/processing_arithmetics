from __future__ import division

import data

import argparse, os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
import numpy as np
import pickle
from keras.models import model_from_json


def shuffleData(d):
    indices = np.arange(len(d[0]))
    np.random.shuffle(indices)
    return zip(*[(d[0][i], d[1][i], d[2][i]) for i in indices])


def defineModel(hidden=None, loss='mse'):
    # generate your input layer, this is not actually containing anything,
    input_layer = Input(shape=(2,), name='input')
    # this is the classifier, activation is linear but can be different of course
    if hidden:
        hidden_layer = Dense(hidden['dHidden'], activation=hidden['aHidden'], weights=None, trainable=True,
                             name='hidden')(input_layer)
        classifier = Dense(1, activation='linear', weights=None, trainable=True, name='output')(hidden_layer)
    else:
        classifier = Dense(1, activation='linear', weights=None, trainable=True, name='output')(input_layer)

    # create the model and compile it
    model = Model(input=input_layer, output=classifier)
    model.compile(loss=loss, optimizer='adam',
                  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_squared_prediction_error',
                           'binary_accuracy'])
    return model


def trainModel(model, tdata, vdata, verbose=2, n=50, batch_size=24):
    # train the model, takes 10 percent out of the training data for validation
    return model.fit(x=np.array(tdata[0]), y=np.array(tdata[1]),
                     validation_data=(np.array(vdata[0]), np.array(vdata[1])), batch_size=batch_size, nb_epoch=n,
                     shuffle=True, verbose=verbose)


def saveModel(model, name='something'):
    model.save_weights(name + '_weights.h5', overwrite=True)
    saved_model = open(name + '.json', 'w').write(model.to_json())


def loadModel(model_name, model_weights):
    model = model_from_json(open(model_name).read())
    model.load_weights(model_weights)
    model.compile(optimizer='adam', loss='mse',
                  metrics=['mean_squared_error', 'mean_squared_prediction_error', 'binary_accuracy'])
    return model


def evaluate(model, data, name):
    results = model.evaluate(np.array(data[0]), np.array(data[1]))
    model_metrics = model.metrics_names
    print('Evaluation on ' + name + ' data (' + str(len(data[0])) + ' examples)')
    print('\t'.join(['%s: %f' % (i, j) for i, j in zip(model_metrics, results)]))
    return results


def printModel(model):
    print(model.summary())
    for layer in model.layers:
        config = layer.get_config()
        if type(config) == dict:
            try:
                print config['name'], config['activation']
            except:
                print ''


def plotHistory(history, loss, saveTo=None):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='heldout')
    plt.legend()
    # plt.xticks(xrange(len(hgtistory.history['loss'])))
    plt.xlabel('Epoch')
    plt.ylabel('Loss (' + loss + ')')
    if saveTo:
        plt.savefig(saveTo)
    else:
        plt.show()


def saveResults(results, metrics, filename):
    identifiers = []
    values = []
    for dataset in sorted(results.keys()):
        identifiers += [dataset + '_' + metric for metric in metrics]
        values += results[dataset]
    with open(filename, 'w') as f:
        f.write(','.join(identifiers) + '\n')
        f.write(','.join([str(v) for v in values]) + '\n')

def trainPrediction(args, dataset, exp):
    if args['dHidden'] > 0:
        hidden = {k: args[k] for k in ['dHidden', 'aHidden']}
    else:
        hidden = None

    model = defineModel(hidden=hidden, loss=args['loss'])
    printModel(model)

    trainData = shuffleData((dataset['X_train']['all'], dataset['Y_train']['all'], dataset['strings_train']['all']))
    valData = shuffleData((dataset['X_heldout']['all'], dataset['Y_heldout']['all'], dataset['strings_heldout']['all']))
    history = trainModel(model=model, tdata=trainData, vdata=valData, n=args['nEpochs'], batch_size=args['bSize'])

    plotHistory(history, args['loss'],os.path.join(args['outDir'], exp + 'convergence.png'))

    saveModel(model, os.path.join(args['outDir'], exp))

    results = {}
    for kind in ['train', 'heldout']:
        results[kind] = evaluate(model, trainData, kind)

    for lan in sorted(dataset['X_test'].keys()):
        testData = (dataset['X_test'][lan], dataset['Y_test'][lan], dataset['strings_test'][lan])
        results['test_' + lan] = evaluate(model, testData, 'test ' + lan)

    saveResults(results, model.metrics_names, os.path.join(args['outDir'], exp + '_results.csv'))
    saveModel(model, os.path.join(args['outDir'], exp))


def main(args):
    print args

    destination = args['out']
    exp = args['experiment']

    if not os.path.exists(destination):
        os.mkdir(destination)

    dataFile = os.path.join(destination, 'kerasData' + str(args['seed']) + '.pik')
    if not os.path.exists(dataFile):
        dataset = data.convert4Keras(args['thetaFile'], seed=args['seed'])
        with open(dataFile, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        print 'Retrieving earlier created data'
        with open(dataFile, 'rb') as f:
            data = pickle.load(f)

    trainPrediction(args, dataset, exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    # data:

    parser.add_argument('-theta', '--thetaFile', type=str, help='File with pickled Theta', required=False)
    parser.add_argument('-exp', '--experiment', type=str, help='Identifier of the experiment', required=True)
    parser.add_argument('-o', '--out', type=str, help='Output name to store model', required=True)
    parser.add_argument('-s', '--seed', type=int, help='Random seed to be used', required=True)
    parser.add_argument('-n', '--nEpochs', type=int, help='Number of epochs to train', required=True)
    parser.add_argument('-b', '--bSize', type=int, default=24, help='Batch size for minibatch training', required=False)
    parser.add_argument('-dh', '--dHidden', type=int, default=0, help='Size of hidden layer', required=False)
    parser.add_argument('-ah', '--aHidden', type=str, choices=['linear', 'tanh', 'relu'],
                        help='Activation of hidden layer', required=False)
    parser.add_argument('-l', '--loss', type=str, choices=['mse', 'mae'],
                        help='Loss function to minimize', required=True)

    args = vars(parser.parse_args())

    main(args)
