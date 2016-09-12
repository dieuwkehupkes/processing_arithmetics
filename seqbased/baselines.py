from generate_training_data import generate_treebank
import numpy as np
import random
import matplotlib.pyplot as plt

def compute_baseline(languages, digits, random_range):

    # generate treebanks
    treebank1 = generate_treebank(languages, digits=digits)
    random.shuffle(treebank1.examples)
    treebank2 = generate_treebank(languages, digits=digits)
    random.shuffle(treebank1.examples)

    # initialise accuracy
    accuracy = 0

    for example1, example2 in zip(treebank1.examples, treebank2.examples):
        expr1, answ1 = example1
        expr2, answ2 = example2

        # generate number within certain range of outckkome of example
        gen_answ1 = np.random.random_integers(answ1-random_range, answ1+random_range, 1)[0]
        gen_answ2 = np.random.random_integers(answ2-random_range, answ2+random_range, 1)[0]

        true_answ = np.argmax([answ1 < answ2, answ1 == answ2, answ1 > answ2])
        gen_answ = np.argmax([gen_answ1 < gen_answ2, gen_answ1 == gen_answ2, gen_answ1 > gen_answ2])

        accuracy += true_answ == gen_answ

    acc = float(accuracy)/len(treebank1.examples)

    return acc

if __name__ == '__main__':
    languages_train = {'L1':10000, 'L2': 10000, 'L4':10000, 'L6':10000}
    languages_test = {'L3': 10000, 'L5':10000, 'L7':10000}
    digits = np.arange(-10,11)
    range = 10

    accuracies_train, accuracies_test = [], []

    ranges = np.arange(0, 30)

    for r in ranges:
        print(r)
        accuracies_train.append(compute_baseline(languages_train, digits, r))
        accuracies_test.append(compute_baseline(languages_test, digits, r))


    plt.plot(ranges, accuracies_train, label="Accuracy training set")
    plt.plot(ranges, accuracies_test, label="Accuracy test set")
    plt.legend()
    plt.show()

