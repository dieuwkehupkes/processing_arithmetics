from keras.models import Sequential
from keras.layers import SimpleRNN
import numpy as np

# generate input data

# alphabet
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
E = np.array([0, 0, 1])
null = np.array([0,0,0])

# define padding function
def padd(array, length=10):
    l = len(array)
    l2 = len(array[0])
    padded_array = np.concatenate((np.zeros((length-l,l2)), array))
    return padded_array

s1 = padd(np.array([a,b]))
s2 = padd(np.array([a,a,b,b]))
s3 = padd(np.array([a,a,a,b,b,b]))
s4 = padd(np.array([a,a,a,a,b,b,b,b]))
s5 = padd(np.array([a,a,a,a,a,b,b,b,b,b]))
s6 = padd(np.array([a,a,b,a,b,b]))

# weight matrices
U = np.array([[0.5, -5, -5], [-5, -1, 5],[-5, -5, -5]])
V = np.array([[0.5, 2, 0], [0, 2, 0], [0, 0, 0]])
out = np.array([0,0,0])
weights = [U, V, out]

# create model
model = Sequential()
model.add(SimpleRNN(3, input_dim=3, input_length=10, return_sequences=True, activation='relu', weights=weights))
model.compile(loss='mean_squared_error', optimizer='sgd')

outputs = model.predict(np.array([s5]), batch_size=1, verbose=1)
print(outputs)
