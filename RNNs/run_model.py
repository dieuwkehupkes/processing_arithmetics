from keras.models import model_from_json
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("architecture", help='JSON file with architecture of the model')
parser.add_argument("weights", help='h5 file with model weights')
parser.add_argument("validation_set", help='pickled validation set')
parser.add_argument("--optimizer", help="optimizer to compile model", default="adagrad")
parser.add_argument("--loss", help="loss function", default="mse")
parser.add_argument("--metrics", help='provide metrics to be monitored during training', default='mspe')

args = parser.parse_args()

# load model
model = model_from_json(open(args.architecture).read())
model.load_weights(args.weights)
model.compile(optimizer=args.optimizer, loss=args.loss, metrics=[args.metrics])


X_val, Y_val, N_digits, N_operators, d_map = pickle.load(open(args.validation_set, 'rb'))

print model.metrics_names
print model.evaluate(X_val, Y_val)