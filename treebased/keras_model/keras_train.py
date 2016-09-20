# imports
from keras.models import Model
from keras.layers import Dense, Input
import numpy as np

X_train = np.array([[-0.96126048,  0.00260371], [-0.99087481, -0.09297248], [0.57209403, -0.99595716], [-0.07151364, -0.99830101], [0.44327486, -0.99749463], [-0.40513734, -0.99966289], [-0.9985929,  0.2795637], [-0.97699455, -0.19296128], [-0.68975031, -0.99909669], [-0.99107905, -0.1580624], [0.27802325, -0.99898107], [-0.11759126, -0.99981301], [-0.99381046,  0.38726686], [-0.9983553 , -0.35453988], [-0.39716189, -0.99916855], [-0.99917485,  0.13008876], [0.5706728 , -0.99883182], [0.55988649, -0.99891811], [-0.743377  , -0.99901815], [-0.99453301, -0.56129514], [-0.09105125, -0.9930295], [-0.99107325,  0.03993487], [-0.99390627,  0.51923931], [0.1713253 , -0.99490251], [-0.04247262, -0.99909119], [0.259236  , -0.99296754], [-0.21412103, -0.99796179], [-0.99621078,  0.27080084], [-9.96220056e-01,   8.89653559e-04], [-0.98476688, -0.36700071], [-0.98551647,  0.63976448], [-0.97704884,  0.00532429], [-0.03900527, -0.99813139], [0.34600109, -0.9992635], [-0.99709804,  0.41672688], [0.35827801, -0.99875548], [0.08328652, -0.99610319], [-0.9939079, -0.0894836], [-0.98903199, -0.19681495], [-0.99072129,  0.68425448], [-0.99211757,  0.38032997], [-0.3102293 , -0.99975892], [-0.99613578, -0.53182784], [-0.01580315, -0.9998212], [-0.27627122, -0.99935522]])

Y_train = np.array([-1, 3, -24, 2, -16, 15, -10, 6, 29, 5, -9, 5, -13, 13, 13, -5, -20, -20, 33, 21, 4, -1, -19, -6, 1, -10, 7, -9, 1, 12, -28, -1, 2, -11, -15, -12, -3, 3, 6, -32, -13, 13, 19, 1, 10])


# generate your input layer, this is not actually containing anything, 
input_layer = Input(shape=(2,), name='input')

# this is the classifier, activation is linear but can be different of course
classifier = Dense(1, activation='linear', weights=None, trainable=True, name='output')(input_layer)

# create the model and compile it
model = Model(input=input_layer, output=classifier)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_prediction_error'])

# train the model, takes 10 percent out of the training data for validation
model.fit({'input': X_train}, {'output': Y_train}, validation_split=0.1, batch_size=24, nb_epoch=200, shuffle=True)



# om een bestaand model te testen:
# model.evaluate(X_test, Y_test)
