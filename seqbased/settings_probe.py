import numpy as np
from auxiliary_functions import max_length

pref = 'models/GRU_A1_2'
model_architecture = pref+'.json'
model_weights = pref+'_weights.h5'
dmap = pref+'.dmap'

classifiers         = ['grammatical', 'intermediate_locally']
# classifiers         = ['intermediate_locally']
# classifiers         = ['grammatical']
nb_epochs           = 10

optimizer           = 'adam'
dropout_recurrent   = 0.0
batch_size          = 24
verbosity           = 2

validation_split    = 0.1

digits              = np.arange(-10,11)

languages_train             = {'L1':3000, 'L2': 3000, 'L4':3000, 'L6':3000}
# languages_val               = {'L3': 400, 'L5':400, 'L7':400}
# languages_val               = None
languages_test              = {'L3': 500, 'L5':500, 'L7':500, 'L7_left':500}
test_separately             = True
maxlen                      = max_length(15)

