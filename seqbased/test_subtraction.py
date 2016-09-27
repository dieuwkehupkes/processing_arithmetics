import sys
from keras.models import model_from_json, Model
from architectures import Training, Probing
import numpy as np
import pickle

np.random.seed(0)

model_architecture = sys.argv[1]
model_weights = sys.argv[2]
dmap = pickle.load(open('models/dmap', 'rb'))
dmap['x'] = 0
dmap_inverted = dict([(item[1],item[0]) for item in dmap.items()])

test_languages = {'L5': 10}
digits = np.arange(-10,11)

test_data = Training.generate_test_data(Probing, test_languages, dmap=dmap, digits=digits,
        classifiers=['subtracting', 'intermediate_locally'], pad_to=57, test_separately=True)

model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


# outputs = [model.layers[6].get_output_at(0), model.layers[4].get_output_at(0)]
outputs = [model.layers[6].get_output_at(0), model.layers[4].get_output_at(0)]

new_model = Model(input=model.layers[0].get_input_at(0), output=outputs)
new_model.compile(loss={'subtracting':'binary_crossentropy', 'intermediate_locally': 'mse'},
        metrics={'subtracting': 'binary_accuracy', 'intermediate_locally': 'mse'}, optimizer='adam', sample_weight_mode='temporal')

for name, X_test, Y_test in test_data:
    i = 0
    sample_weights = {}
    for key in Y_test:
        sample_weight = np.zeros_like(X_test['input'])
        sample_weight[X_test['input']!=0] = 1
        sample_weights[key] = sample_weight

    print '\t'.join(["%s: %f" % (name, outcome) for name, outcome in zip(new_model.metrics_names, new_model.test_on_batch(X_test, Y_test, sample_weight=sample_weights))])
    predictions = new_model.predict(X_test)
    for s, y_sub, y_outc  in zip(X_test['input'], Y_test['subtracting'], Y_test['intermediate_locally']):
        m = {'subtracting': y_sub, 'intermediate_locally': y_outc}
        # print "accuracy for cur example:"
        # print '\t'.join(["%s: %f" % (name, outcome) for name, outcome in zip(new_model.metrics_names[3:], new_model.evaluate(X_test, Y_test)[3:])])
        nzi = s.nonzero()[0][0]
        labels = [dmap_inverted[word] for word in s[nzi:]]
        sub_target = [sub[0] for sub in y_sub[nzi:]]
        sub_target_all = [sub[0] for sub in y_sub]
        outcome_target = [sub[0] for sub in y_outc[nzi:]]
        outcome_target_all = [sub[0] for sub in y_outc]
        sub_model = [int(round(sub[0])) for sub in predictions[0][i][nzi:]]
        sub_model_all = [int(round(sub[0])) for sub in predictions[0][i]]
        outcome_model = [sub[0] for sub in predictions[1][i][nzi:]]
        outcome_model_all = [sub[0] for sub in predictions[1][i]]
        outcome_model_rounded = [int(round(sub[0])) for sub in predictions[1][i][nzi:]]
        print '\nseq: ', '  '.join(labels)
        print 'sub target:\t\t', '  '.join([str(x) for x in sub_target])
        print 'sub model:\t\t', '  '.join([str(x) for x in sub_model])
        # print 'sub target all:\t\t', '  '.join([str(x) for x in sub_target_all])
        # print 'sub model all:\t\t', '  '.join([str(x) for x in sub_model_all])

        print np.array(sub_target) == np.array(sub_model)

        # print 'model sub all:\t', '  '.join([str(x) for x in sub_model_all])
        # print 'model outcome all:\t', '  '.join([str(x) for x in outcome_model])

        print '\n\n\n'

        
        print '\noutc target:\t\t', '  '.join([str(x) for x in outcome_target])
        print 'outc model:\t\t', '  '.join([str(x) for x in outcome_model_rounded])
        # print np.mean(np.power(np.array(outcome_target_all)-np.array(outcome_model_all), 2))
        # print np.mean(np.power(np.array(outcome_target)-np.array(outcome_model), 2))
        #print 'model rounded:\t', '  '.join([str(x) for x in outcome_model_rounded])
        # print 'model rounded all:\t', '  '.join([str(x) for x in outcome_model_all])
        # print '\n model predictions:'
        i += 1
        raw_input()

