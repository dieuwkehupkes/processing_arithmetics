import sys
from keras.layers import Dense, TimeDistributed
from keras.models import load_model
from processing_arithmetics.sequential.architectures import DiagnosticClassifier
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy as np
import pickle
import scipy

# np.random.seed(0)

dc = DiagnosticClassifier(model=sys.argv[1], classifiers=['intermediate_locally', 'intermediate_recursively', 'subtracting'])

dc_model = load_model(sys.argv[1])

test_languages = OrderedDict(zip(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L9', 'L9R', 'L9L'], [[100]*11]))
digits = np.arange(-10,11)

test_data = dc.generate_test_data(test_languages, digits=digits)

for name, X_test, Y_test in test_data:
    i = 0
    predictions = dc_model.predict(X_test)
    print "Compute predictions for", name
    for s, y_sub, y_imm, y_recurs in zip(X_test['input'], Y_test['subtracting'], Y_test['intermediate_locally'], Y_test['intermediate_recursively']):

        # targets
        m = {'subtracting': y_sub, 'intermediate_locally': y_imm, 'intermediate_recursively': y_recurs}
        # non-zero indices
        nzi = s.nonzero()[0][0]

        xticks = [dmap_inverted[word] for word in s[nzi:]]

        # generate targets
        sub_target = [sub[0] for sub in y_sub[nzi:]]
        recursive_target = [sub[0] for sub in y_recurs[nzi:]]
        imm_target = [sub[0] for sub in y_imm[nzi:]]

        # model outcomes
        sub_model = [sub[0] for sub in predictions[0][i][nzi:]]
        sub_model_rounded = [int(round(sub[0])) for sub in predictions[0][i][nzi:]]
        imm_model = [sub[0] for sub in predictions[1][i][nzi:]]
        imm_model_rounded = [int(round(sub[0])) for sub in predictions[1][i][nzi:]]
        recursive_model = [sub[0] for sub in predictions[2][i][nzi:]]
        recursive_model_rounded = [int(round(sub[0])) for sub in predictions[2][i][nzi:]]
        original_pred = [sub[0] for sub in predictions[3][i][nzi:]]
        original_pred_rounded = [int(round(sub[0])) for sub in predictions[3][i][nzi:]]

        rec_targets.append(recursive_target)
        imm_targets.append(imm_target)
        rec_model_all.append(recursive_model_rounded)
        imm_model_all.append(imm_model_rounded)

        
        i += 1

    rec_gold = np.array(rec_targets).ravel()
    rec_model = np.array(rec_model_all).ravel()
    imm_gold = np.array(imm_targets).ravel()
    imm_model = np.array(imm_model_all).ravel()

    print "correlation recursive trajectories:", scipy.stats.pearsonr(rec_gold, rec_model)
    print "correlation incremental trajectories:", scipy.stats.pearsonr(imm_gold, imm_model)

