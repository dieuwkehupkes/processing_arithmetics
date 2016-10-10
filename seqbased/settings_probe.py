import numpy as np
from auxiliary_functions import max_length
from arithmetics import training_treebank, test_treebank, heldout_treebank


model_seed = 2
model = 'models_run2/GRU15postfix_seed'+str(model_seed)+'_1500.h5' 
dmap = 'models/dmap'
save_every = False

seed                = 20
seed_test           = 100
classifiers         = ['intermediate_recursively']
nb_epochs           = 500

filename = 'models_probe/GRU15postfix_seed'+str(model_seed)+'probe_seed'+str(seed)+'_'
raw_input(filename)

optimizer           = 'adam'
dropout_recurrent   = 0.0
batch_size          = 24
verbosity           = 2

validation_split    = 0.1
format              = 'postfix'

digits              = np.arange(-10,11)

# languages_train             = {'L1':3000, 'L2': 3000, 'L4':3000, 'L6':3000}
languages_train               = training_treebank(seed=seed)
languages_val               = heldout_treebank(seed=seed)
# languages_val               = {'L3': 400, 'L5':400, 'L7':400}
languages_test              = [(name, treebank) for name, treebank in test_treebank(seed=seed_test)]
sample_weights              = None
test_separately             = True
maxlen                      = max_length(15)
mask_zero                   = True

