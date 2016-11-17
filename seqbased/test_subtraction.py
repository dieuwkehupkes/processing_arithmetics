import sys
from keras.layers import Dense, TimeDistributed
from keras.models import load_model, Model
from architectures import Training, Probing
import matplotlib.pylab as plt
import numpy as np
import pickle
import scipy

seed = 0

# np.random.seed(0)

model = sys.argv[1]       # probe model
model_probe = model[:-8]+'probe_seed10_500.h5'

dmap = pickle.load(open('models/dmap', 'rb'))
dmap['x'] = 0
dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])

test_languages = {'L9': 5000}
digits = np.arange(-10,11)

test_data = Training.generate_test_data(Probing, test_languages, dmap=dmap, digits=digits, 
                                        classifiers=['subtracting', 'intermediate_locally', 'intermediate_recursively'],
                                        pad_to=57, test_separately=True)

expr = '(  (  (  (  -10  -  -7  )  -  (  -7  -  -3  )  )  +  8  )  -  (  (  2  -  (  -9  -  -4  )  )  +  4  )  )'


model_probe = load_model(model_probe)
model = load_model(model)
original_weights = model.layers[-1].get_weights()

original_prediction = TimeDistributed(Dense(1, weights=original_weights, activation='linear'), name='original_prediction')(model_probe.layers[4].output)

outputs = [model_probe.layers[5].get_output_at(0), model_probe.layers[6].get_output_at(0), model_probe.layers[7].get_output_at(0), original_prediction]

new_model = Model(input=model_probe.layers[0].get_input_at(0), output=outputs)
new_model.compile(loss={'subtracting':'binary_crossentropy', 'intermediate_locally': 'mse', 'original_prediction': 'mse', 'intermediate_recursively': 'mse'},
                  metrics={'subtracting': 'binary_accuracy', 'intermediate_locally': 'mse', 'original_prediction':'mse', 'intermediate_recursively':'mse'},
                  optimizer='adam', sample_weight_mode='temporal')

rec_targets, imm_targets, rec_model_all, imm_model_all = [], [], [], []

for name, X_test, Y_test in test_data:
    i = 0
    predictions = new_model.predict(X_test)
    print new_model.output_names
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

        

        # print '\nseq: ', '  '.join(xticks)
        # print 'sub target:\t\t', '  '.join([str(x) for x in sub_target])
        # print 'sub model:\t\t', '  '.join([str(x) for x in sub_model])
        # print np.array(sub_target) == np.array(sub_model_rounded)

        # print '\noutc target:\t\t', '  '.join([str(x) for x in outcome_target])
        # print 'outc model:\t\t', '  '.join([str(x) for x in outcome_model])

        # plot results
        # fig = plt.figure()

        # ranges = xrange(len(xticks))

        # fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[1, 3]})
        # ax1.plot(ranges, sub_target, label='subtraction target', color='g', ls='--')
        # ax1.plot(ranges, sub_model, label='subtraction model', color='g', linewidth=2)
        # ax1.axhline(0.5, color='black')
        # ax1.set_xticks(ranges)
        # ax1.set_xticklabels(xticks)
        # ax1.set_ylim([-0.1, 1.1])
        # ax1.legend(bbox_to_anchor=(.35, 1.85))

        # # ax = fig.add_subplot(2, 1, gridspec_kw = {'width_ratios':[2, 1]})

        # ax2.plot(ranges, original_pred, label='model', linewidth=2, color='r')
        # ax2.plot(ranges, imm_target, label='immStrat target', color='g', ls='--')
        # # ax2.plot(ranges, recursive_target, label='recStrat target', color='b', ls='--')

        # ax2.plot(ranges, imm_model, label='immStrat model', color='g')
        # # ax2.plot(ranges, recursive_model, label='recStrat model', color='b')

        # ax2.set_xticks(ranges)
        # ax2.set_xticklabels(xticks)
        # ax2.legend(bbox_to_anchor=(.7, 2.0))
        # plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.75)
        # plt.show()

        i += 1

    rec_gold = np.array(rec_targets).ravel()
    rec_model = np.array(rec_model_all).ravel()
    imm_gold = np.array(imm_targets).ravel()
    imm_model = np.array(imm_model_all).ravel()

    print "correlation recursive trajectories:", scipy.stats.pearsonr(rec_gold, rec_model)
    print "correlation incremental trajectories:", scipy.stats.pearsonr(imm_gold, imm_model)

