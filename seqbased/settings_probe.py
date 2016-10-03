import numpy as np
from auxiliary_functions import max_length
from arithmetics import training_treebank, test_treebank, heldout_treebank

model = 'models_run2/GRU15_seed3_1500.h5' 
dmap = 'models/dmap'
save_every = 800

seed                = 10
seed_test           = 100
classifiers         = ['grammatical', 'intermediate_locally', 'intermediate_recursively', 'subtracting']
nb_epochs           = 800

filename = model+'_probe_seed'+str(seed)+'.h5'

optimizer           = 'adam'
dropout_recurrent   = 0.0
batch_size          = 24
verbosity           = 2

validation_split    = 0.1
format              = 'infix'

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

