import numpy as np
from auxiliary_functions import max_length

pref = 'models/GRU_A1_2'
model_architecture = pref+'.json'
model_weights = pref+'_weights.h5'
dmap = pref+'.dmap'

classifiers         = ['grammatical']
nb_epoch            = 3000

optimizer           = 'adam'
dropout_recurrent   = 0.0
batch_size          = 24
nb_epochs           = 3
verbosity           = 2

validation_split    = 0.1

digits              = np.arange(-10,11)

languages_train             = {'L1':1, 'L2': 2, 'L4':1, 'L6':1}
# languages_train             = {'L1':3000, 'L2': 3000, 'L4':3000, 'L6':3000}
languages_val               = None
languages_test              = {'L3': 400, 'L5':400, 'L7':400}
maxlen                      = max_length(15)

