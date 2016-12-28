import numpy as np
import sys
sys.path.insert(0, '../commonFiles') 
from arithmetics import mathTreebank
import random
import matplotlib.pyplot as plt

def compute_baseline(treebank, digits, random_range):

    # initialise accuracy
    sse = 0
    sae = 0
    correct = 0
    correct_single = 0
    

    for expr1, expr2, comp in treebank.pairedExamples:

        ans1 = expr1.solve()
        ans2 = expr2.solve()

        # generate number within certain range of outckkome of example
        gen_ans1 = random_range*(np.random.ranf()-0.5) + ans1
        gen_ans2 = random_range*(np.random.ranf()-0.5) + ans2


        sae += abs(gen_ans1-ans1)
        sse += np.power(ans1- gen_ans1, 2)
        correct_single += ans1 == np.round(gen_ans1)

        gen_c = np.argmax([gen_ans1 < gen_ans2, gen_ans1 == gen_ans2, gen_ans1 > gen_ans2])
        gen_comp = {0:'<', 2:'>', 1:'='}

        correct += gen_comp == comp

    total = float(len(treebank.examples))

    accuracy = correct/total
    mse = sse/total
    mae = sae/total
    binary_accuracy = correct_single/total

    return accuracy, mse, mae, binary_accuracy

if __name__ == '__main__':

    languages_test = dict([('L1', 50), ('L2', 100), ('L3', 150), ('L4', 300), ('L5', 500), ('L6', 1000), ('L7', 1500), ('L8', 1500), ('L9', 1500), ('L9_left', 1500)])


    digits = np.arange(-10,11)

    treebank = mathTreebank(languages_test, digits=digits)

    accs, mses, maes, bin_accs = [], [], [], []

    ranges = np.arange(0, 10)

    for r in ranges:
        print r
        acc, mse, mae, binary_accuracy = compute_baseline(treebank, digits, r)
        accs.append(acc)
        mses.append(mse)
        maes.append(mae)
        bin_accs.append(binary_accuracy)


    print("start plotting")

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(ranges, bin_accs)
    ax1.title.set_text("Binary accuracy compare")
    ax2 = fig.add_subplot(222)
    ax2.plot(ranges, bin_accs)
    ax2.title.set_text("Binary accuracy scalar prediction")
    ax3 = fig.add_subplot(223)
    ax3.plot(ranges, mses)
    ax3.title.set_text("Mean Squared Error")
    ax4 = fig.add_subplot(224)
    ax4.plot(ranges, maes)
    ax4.title.set_text("Mean Absolute Error")
    plt.xlabel("range")
    plt.show()
    exit()

