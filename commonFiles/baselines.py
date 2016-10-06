import numpy as np
import sys
sys.path.insert(0, '../commonFiles') 
from arithmetics import mathTreebank, test_treebank
from collections import OrderedDict
import random
import matplotlib
import matplotlib.pyplot as plt

def compute_baseline_range(treebank, random_range, distr='normal'):

    # initialise accuracy
    sse = 0
    sae = 0
    correct = 0
    correct_single = 0
    

    for expr1, expr2, comp in treebank.pairedExamples:

        ans1 = expr1.solve()
        ans2 = expr2.solve()

        # generate number within certain range of outckkome of example
        if distr == 'normal':
            gen_ans1 = np.random.normal(loc=ans1, scale=random_range) 
            gen_ans2 = np.random.normal(loc=ans2, scale=random_range) 
        elif distr == 'uniform':
            gen_ans1 = np.random.uniform(low=ans1-0.5*random_range, high=ans1+0.5*random_range) 
            gen_ans2 = np.random.uniform(low=ans2-0.5*random_range, high=ans2+0.5*random_range) 
        else:
            print "non valid distribution"


        sae += abs(gen_ans1-ans1)
        sse += np.square(ans1 - gen_ans1)
        correct_single += ans1 == np.round(gen_ans1)

        gen_comp = np.argmax([gen_ans1 < gen_ans2, gen_ans1 == gen_ans2, gen_ans1 > gen_ans2])
        gen_comp = {0:'<', 1:'=', 2:'>'}[gen_comp]

        correct += gen_comp == comp

    total = float(len(treebank.examples))

    accuracy = correct/total
    mse = sse/total
    mae = sae/total
    binary_accuracy = correct_single/total

    return accuracy, mse, mae, binary_accuracy

def compute_baseline_noise(treebank, digit_noise, operator_noise):

    # initialise accuracy
    sse = 0
    sae = 0
    correct = 0
    correct_single = 0
    

    for expr1, expr2, comp in treebank.pairedExamples:

        ans1 = expr1.solve()

        # generate number within certain range of outckkome of example
        gen_ans1 = eval(expr1.toString(digit_noise=digit_noise, operator_noise=operator_noise))
        gen_ans2 = eval(expr2.toString(digit_noise=digit_noise, operator_noise=operator_noise))
        # gen_ans1 = np.random.normal(loc=ans1, scale=random_range)
        # gen_ans2 = np.random.normal(loc=ans2, scale=random_range)


        sae += abs(gen_ans1-ans1)
        sse += np.square(ans1 - gen_ans1)
        correct_single += ans1 == np.round(gen_ans1)

        gen_comp = np.argmax([gen_ans1 < gen_ans2, gen_ans1 == gen_ans2, gen_ans1 > gen_ans2])
        gen_comp = {0:'<', 1:'=', 2:'>'}[gen_comp]

        correct += gen_comp == comp

    total = float(len(treebank.examples))

    accuracy = correct/total
    mse = sse/total
    mae = sae/total
    binary_accuracy = correct_single/total

    return accuracy, mse, mae, binary_accuracy

if __name__ == '__main__':

    # matplotlib.rcParams['xtick.labelsize'] = 8
    languages_test = OrderedDict([('L9_left', 1500), ('L9_right', 1500), ('L1', 50), ('L2', 100), ('L3', 150), ('L4', 300), ('L5', 500), ('L6', 1000), ('L7', 1500), ('L8', 1500), ('L9', 1500)])

    digits = np.arange(-10,11)

    languages = test_treebank(seed=5, languages=languages_test) 
    operator_noise = 0.1
    digit_noise = 0.5

    accs, mses, maes, bin_accs = [], [], [], []

    for name, treebank in languages:
        # acc, mse, mae, binary_accuracy = compute_baseline_noise(treebank, digit_noise=digit_noise, operator_noise=operator_noise)
        acc, mse, mae, binary_accuracy = compute_baseline_range(treebank, random_range=3)
        accs.append(acc)
        mses.append(mse)
        maes.append(mae)
        bin_accs.append(binary_accuracy)

    xticks = [l[1:] for l in languages_test.keys()]
    xticks[0]='9L'
    xticks[1]='9R'
    ranges = 0.15*np.arange(len(xticks))
    width=0.1

    noise_pars = "Operator noise = %f, Digit noise = %f" % (operator_noise, digit_noise) 

    fig = plt.figure()
    axes = {} 
    for nr in xrange(1,5):
        axes[nr] = fig.add_subplot(2,2,nr)
        axes[nr].set_xticks(ranges)
        axes[nr].set_xticklabels(xticks)

    axes[1].title.set_text("Binary accuracy compare")
    axes[2].title.set_text("Binary accuracy scalar prediction")
    axes[2].set_ylim([0, 1])
    axes[3].title.set_text("Mean Squared Error")
    axes[4].title.set_text("Mean Absolute Error")
    
    axes[1].bar(ranges, accs, width=width, align='center')
    axes[2].bar(ranges, bin_accs, width=width, align='center')
    axes[3].bar(ranges, mses, width=width, align='center')
    axes[4].bar(ranges, maes, width=width, align='center')

    plt.show()

