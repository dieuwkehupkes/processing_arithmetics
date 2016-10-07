# imports
from keras.models import Model
from keras.layers import Dense, Input
import numpy as np
import pickle
from keras.models import model_from_json
from collections import defaultdict
import sys
import argparse

def getData(directory = '../data4keras'):
  import data4keras.trainData02 as trainData
  import data4keras.testData02 as testData
#  trainData = []
#  testData = []
##  for name in ['X_','Y_','strings_']:
 #   with open(directory + '/' +name + 'train' + '.pkl', 'rb') as f:
 #       trainData.append(pickle.load(f))
 #   with open(directory + '/' +name + 'test' + '.pkl', 'rb') as f:
 #       testData.append(pickle.load(f))
  return (trainData.inputs,trainData.outputs,trainData.strings),(testData.inputs,testData.outputs,testData.strings)


def shuffleData(d):
  indices = np.arange(len(d[0]))
  np.random.shuffle(indices)
  return zip(*[(d[0][i],d[1][i],d[2][i]) for i in indices])

def defineModel(hidden=None):
	# generate your input layer, this is not actually containing anything, 
	input_layer = Input(shape=(2,), name='input')

	# this is the classifier, activation is linear but can be different of course
	if hidden: 
          hidden_layer = Dense(hidden['dHidden'], activation=hidden['aHidden'], weights=None, trainable=True, name='hidden')(input_layer)
	  classifier = Dense(1, activation='linear', weights=None, trainable=True, name='output')(hidden_layer)
	else:
          classifier = Dense(1, activation='linear', weights=None, trainable=True, name='output')(input_layer)

	# create the model and compile it
	model = Model(input=input_layer, output=classifier)
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error','mean_squared_prediction_error','binary_accuracy'])

	return model

def trainModel(model,data, verbose = 2,n=50, batch_size=24):
	# train the model, takes 10 percent out of the training data for validation
	history = model.fit({'input': np.array(data[0])}, {'output': np.array(data[1])}, validation_split=0.1, batch_size=batch_size, nb_epoch=n, shuffle=True, verbose = verbose)

def saveModel(model, name = 'something'):
	model.save_weights(name+'_weights.h5',overwrite=True)
	saved_model = open(name+'.json', 'w').write(model.to_json())

def loadModel(model_name):
	model = model_from_json(open(model_name).read())
	model.load_weights(model_weights)
	model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error','mean_squared_prediction_error','binary_accuracy'])
	return model

def evaluate(model, data, name):
  model_metrics = model.metrics_names
  results = model.evaluate(np.array(data[0]), np.array(data[1]))
  print('Evaluation on '+name+' data ('+str(len(data[0]))+' examples)')
  print('\t'.join(['%s: %f' % (i,j) for i, j in zip(model_metrics,  results)]))
  dataPerLength = defaultdict(list)
  for item, label, string in zip(*data):
    expressionL = (len(string.split())+3)/4
    dataPerLength[expressionL].append((item,label))
  perL = {}
  for length, databit in dataPerLength.iteritems():
    x,y = zip(*databit)
    results=model.evaluate(np.array(x),np.array(y))
    print('results for length '+str(length)+':' +'\t'.join(['%s: %f' % (i,j) for i, j in zip(model_metrics,  results)]))
    #perL[length]=results[model_metrics.index('mean_squared_prediction_error')]
#  print('\tMSPE per length: '+ str(perL))

def printModel(model):
  print(model.summary())
  for layer in model.layers:
    config  = layer.get_config()
    if type(config)  == dict:
      try: print config['name'],config['activation']
      except: print ''


def main(args):
  name  = args['out']
  trainData,testData = getData()
  trainData = shuffleData(trainData)
  testData = shuffleData(testData)


  if args['dHidden']>0: hidden = {k:args[k] for k in ['dHidden','aHidden']}
  else: hidden = None

  model=defineModel(hidden=hidden)
  printModel(model)

  trainModel(model,trainData,n=args['nEpochs'], batch_size=args['bSize'])
  saveModel(model, name)
  evaluate(model,testData,'test')
  evaluate(model,trainData,'train')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
 # data:
  parser.add_argument('-exp','--experiment', type=str, help='Identifier of the experiment', required=True)
  parser.add_argument('-o','--out', type=str, help='Output name to store model', required=True)
#  parser.add_argument('-s', '--seed', type=int, help='Random seed to be used', required=True)
  parser.add_argument('-n','--nEpochs', type=int, help='Number of epochs to train', required=True)
  parser.add_argument('-b','--bSize', type=int, default = 24, help='Batch size for minibatch training', required=False)
  parser.add_argument('-dh','--dHidden', type=int, default=0,help='Size of hidden layer', required=False)
  parser.add_argument('-ah','--aHidden', type=str, choices=['linear','tanh','relu'], help='Activation of hidden layer', required=False)


  args = vars(parser.parse_args())

  main(args)
