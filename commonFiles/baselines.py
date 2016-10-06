import numpy as np
import sys
sys.path.insert(0, '../commonFiles') 
from arithmetics import mathTreebank, test_treebank
from collections import OrderedDict
import random
import matplotlib.pyplot as plt

def compute_baseline(treebank, digit_noise, operator_noise):

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

    languages_test = OrderedDict([('L1', 50), ('L2', 100), ('L3', 150), ('L4', 300), ('L5', 500), ('L6', 1000), ('L7', 1500), ('L8', 1500), ('L9', 1500), ('L9_left', 1500)])

    digits = np.arange(-10,11)

    languages = test_treebank(seed=5, languages=languages_test)

    operator_noise = 0.2
    digit_noise = 0.5

    accs, mses, maes, bin_accs = [], [], [], []

    for name, treebank in languages:
        acc, mse, mae, binary_accuracy = compute_baseline(treebank, digit_noise=digit_noise, operator_noise=operator_noise)
        accs.append(acc)
        mses.append(mse)
        maes.append(mae)
        bin_accs.append(binary_accuracy)


    print("start plotting")

    ranges = xrange(len(languages_test))
    xticks = languages_test.keys()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(ranges, accs)
    ax1.title.set_text("Binary accuracy compare")
    ax1.set_xticklabels(xticks)
    ax2 = fig.add_subplot(222)
    ax2.plot(ranges, bin_accs)
    ax2.title.set_text("Binary accuracy scalar prediction")
    ax2.set_xticklabels(xticks)
    ax3 = fig.add_subplot(223)
    ax3.plot(ranges, mses)
    ax3.title.set_text("Mean Squared Error")
    ax3.set_xticklabels(xticks)
    ax4 = fig.add_subplot(224)
    ax4.plot(ranges, maes)
    ax4.title.set_text("Mean Absolute Error")
    ax4.set_xticklabels(xticks)
    plt.xlabel("range")
    plt.show()
    exit()

