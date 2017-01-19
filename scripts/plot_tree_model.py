import pickle
import argparse
import data
import numpy as np
from .. import arithmetics
# import core.activation as myActivation

import plotTools as pt

eC = 'blue'
projColors = ['green', 'orange', 'purple', 'blue']


def plot_embs(theta):
    wm = theta[('word',)]
    mbits = np.split(theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')], 3, axis=1)
    fig, ax = pt.new_fig()
    ranges = [pt.colorscale(20, projColors[i]) for i in range(3)]
    for w, v in wm.iteritems():
        if w[-1].isdigit():
            pt.scatter_and_annotate(ax, point=v, label=w, color=eC, pos='right')
            for i in [0, 2]:
                proj = mbits[i].dot(v)
                pt.draw_arrow(ax, v, proj, label='', color=projColors[i], head=0, alpha=0.3)

                if int(w) in [-10, -5, 0, 5, 10]:
                    label = w + ('L' if i == 0 else 'R')
                else:
                    label = ''
                pt.scatter_and_annotate(ax, point=proj, label=label, color=ranges[i](int(w) + 10),
                                      pos=('left' if i == 2 else 'right'))
        else:
            print w
    pt.title('Effect of composition as left or right child on embeddings')
    pt.improve_ticks(ax)
    if outDir:
        pt.save_figure(ax, True, outDir, 'embs')
    else:
        pt.show(ax, True)


def plot_matrices(theta):
    for name, mat in theta.iteritems():
        if 'M' in name:
            pt.showmat(mat, cb=True)
            pt.title(name)
            if outDir:
                pt.save_figure(None, False, outDir, 'matrix_' + name[0])
            else:
                pt.show(None, False)


def plot_for_subtree(ax, nw, theta, n=0):
    M = theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
    mbits = np.split(M, 3, axis=1)
    B = theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'B')]
    children = nw.inputs
    reps = [c.a for c in children]
    projs = [mbits[i].dot(reps[i]) for i in range(3)]
    names = []
    for c in children:
        try:
            names.append(c.key)
        except:
            names.append(str(c))
    names.append('bias')

    # plot Project
    stuff = [pt.scatter_and_annotate(ax, reps[i], eC, names[i]) for i in range(3)]
    stuff += [pt.scatter_and_annotate(ax, projs[i], projColors[i], names[i] + '\'') for i in range(3)]
    stuff += [pt.draw_arrow(ax, reps[i], projs[i], label='project' + ('L' if i == 0 else ('R' if i == 2 else '')),
                           color=projColors[i], alpha=0.3, head=0.1) for i in range(3)]

    n += 1
    if outDir:
        pt.save_figure(ax, False, outDir, 'example' + str(n) + 'Project')
    else:
        pt.show(ax, False)

    [item.remove() for sublist in stuff for item in sublist]
    now = np.array([0, 0])
    stuff = []
    for i, proj in enumerate(projs + [B]):
        nnow = now + proj
        stuff.append(pt.scatter_and_annotate(ax, nnow, projColors[i], ('' if i < 3 else 'sum')))
        stuff.append(pt.draw_arrow(ax, now, nnow, label=names[i] + '\'', color=projColors[i], alpha=0.3, head=0.1))
        now = nnow

    n += 1
    if outDir:
        pt.save_figure(ax, False, outDir, 'example' + str(n) + 'Sum')
    else:
        pt.show(ax, False)
    [item.remove() for sublist in stuff for item in sublist]

    squashed = nw.a  # myActivation.activate(now,'tanh')[0]
    # print nw.a-squashed, 'should be (close to) zero'
    stuff = []
    stuff.append(pt.scatter_and_annotate(ax, now, 'black', 'sum'))
    stuff.append(pt.scatter_and_annotate(ax, squashed, eC, str(nw)))
    stuff.append(pt.draw_arrow(ax, now, squashed, label='squash', color='blue', alpha=0.3, head=0.1))

    n += 1
    if outDir:
        pt.save_figure(ax, False, outDir, 'example' + str(n) + 'Squash')
    else:
        pt.show(ax, False)
    [item.remove() for sublist in stuff for item in sublist]


def plot_for_subtree2(nw, theta, n=0):
    fig, ax = pt.new_fig()
    pt.equal_aspects(ax)
    M = theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
    mbits = np.split(M, 3, axis=1)
    B = theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'B')]
    children = nw.inputs
    reps = [c.a for c in children]
    projs = [mbits[i].dot(reps[i]) for i in range(3)]
    names = []
    for c in children:
        try:
            names.append(c.key)
        except:
            names.append(str(c))
    names.append('bias')

    now = np.array([0, 0])
    for i, proj in enumerate(projs + [B]):
        nnow = now + proj
        pt.scatter_and_annotate(ax, nnow, projColors[i], '', alpha=0)
        if i == 0: postfix = 'L'
        elif i == 1: postfix = 'M'
        elif i == 2: postfix = 'R'
        else: postfix = ''
        pt.draw_arrow(ax, now, nnow, label=names[i] + postfix, color=projColors[i],
                     alpha=0.6, head=0.1)
        now = nnow
    pt.scatter_and_annotate(ax, nnow, projColors[i], '')
    # print nw.a-myActivation.activate(now,'tanh')[0], 'should be (close to) zero'
    # pt.scatter_and_annotate(ax, now , 'black', 'sum')
    pt.scatter_and_annotate(ax, nw.a, eC, str(nw), pos='bot')
    pt.draw_arrow(ax, now, nw.a, label='squash', color='black', alpha=0.6, head=0.1)

    pt.put_origin(ax)
    pt.minimal_ticks(ax)


    if outDir:
        pt.save_figure(ax, False, outDir, 'example' + str(n) + 'All')
    else:
        pt.show(ax, False)


def plot_example(theta):
    fig, ax = pt.new_fig(([-1.5, 3.5], [-3, 4.5]))
    pt.minimal_ticks(ax)
    me = arithmetics.mathExpression.fromstring('( 5 - ( 2 + 3 ) )')
    ans = me.solve()
    print str(me), ans

    nw = data.NN.RNN(me)
    nw.activate(theta)

    # plot_for_subtree(ax, nw.root.inputs[2], theta,0)
    # plot_for_subtree(ax, nw.root, theta, 10)

    plot_for_subtree2(nw.root.inputs[2], theta, 0)
    plot_for_subtree2(nw.root, theta, 10)


def plot_comparison(theta):
    try:
        left, right = np.split(theta[('comparison', 'M')], 2, axis=1)
    except:
        print 'no comparison matrix in theta'
        return

    mtb = arithmetics.training_treebank(args['seed'], languages={'L9': 25}, )
    rnnTB = data.RNNTB(mtb.examples)
    colors = pt.colorscale(120)
    rnn, tar = rnnTB.examples[0]
    rep = rnn.activate(theta)
    d = len(rep)
    for i in range(d):
        for j in range(i + 1, d):

            fig, ax = pt.new_fig()

            for rnn, tar in rnnTB.get_examples():
                rep = rnn.activate(theta)
                leftie = left.dot(rep)
                rightie = right.dot(rep)
                # print leftie.shape, rightie.shape
                # pt.scatter_and_annotate(ax, rep, color=colors(tar+60), label=tar, size=8, txtcolor='black')
                pt.scatter_and_annotate(ax, (leftie[i], leftie[j]), color=colors(tar + 60), label=str(tar) + 'L', size=8,
                                      txtcolor='black')
                pt.scatter_and_annotate(ax, (rightie[i], rightie[j]), color=colors(tar + 60), label=str(tar) + 'R',
                                      size=8, txtcolor='black')

            pt.title('Effect of comparison layer on L9 expressions, dims ' + str(i) + '-' + str(j))
            if outDir:
                pt.save_figure(ax, True, outDir, 'comparison' + str(i) + str(j))
            else:
                pt.show(ax, True)


def plot_answers(theta):
    mtb = data.arithmetics.training_treebank(args['seed'], languages={'L9': 2000}, )
    rnnTB = data.RNNTB(mtb.examples)
    fig, ax = pt.new_fig()
    colors = pt.colorscale(120, 'blue')
    ranges = [pt.colorscale(120, projColors[i]) for i in range(3)]
    mbits = np.split(theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')], 3, axis=1)
    tolabel = {'+': {i: [] for i in range(-60, 61, 10)}, '-': {i: [] for i in range(-60, 61, 10)}}
    # tolabel=tolabel+tolabel[:]+tolabel[:]
    doLater = []
    for rnn, tar in rnnTB.get_examples():
        rep = rnn.activate(theta)
        for i in [0, 2]:
            proj = mbits[i].dot(rep)
            doLater.append((rep, proj, projColors[i], ranges[i](tar + 60)))

        if tar in tolabel['-'].keys():
            tolabel[rnn.root.inputs[1].key][tar].append(rep)
            # pt.scatter_and_annotate(ax, rep, color=colors(tar+60), label=tar, size=10, txtcolor='black')
            # del tolabel[tolabel.index(tar)]
        else:
            pt.scatter_and_annotate(ax, rep, color=colors(int(tar) + 60), label='', txtcolor='black')
    stuff = []
    for op, d in tolabel.iteritems():
        print op
        for tar, replist in d.iteritems():
            if len(replist) == 0:  continue
            av = np.average(np.array(replist), axis=0)
            stuff.append(pt.scatter_and_annotate(ax, av, color=colors(tar + 60), label=tar, txtcolor='black',
                                               pos=('right' if op == '+' else 'top')))
    pt.title('Root representations of L9 expressions')

    if outDir:
        pt.save_figure(ax, True, outDir, 'answers')
    else:
        pt.show(ax, True)

    [item.remove() for sublist in stuff for item in sublist]
    for rep, proj, pColor, color in doLater[:200]:
        pt.draw_arrow(ax, rep, proj, label='', color=pColor, head=0, alpha=0.3)
        pt.scatter_and_annotate(ax, point=proj, label='', color=color)
    pt.title('Root representations and Projections of L9 expressions')
    if outDir:
        pt.save_figure(ax, True, outDir, 'answersProj')
    else:
        pt.show(ax, True)


def main(args):
    with open(args['theta'], 'rb') as f:
        theta = pickle.load(f)

    global outDir
    if args['outDir'] == 'print':
        outDir = None
    else:
        outDir = args['outDir']
    # plot_matrices(theta)
    # plot_embs(theta)
    plot_example(theta)
    # plot_comparison(theta)
    # plot_answers(theta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('-t', '--theta', type=str,
                        default='trainedModels/trainedModels2/304/phase0startEpoch1000.theta.pik',
                        help='File with pickled theta', required=False)
    parser.add_argument('-o', '--outDir', type=str, default='../../ArithmeticsPaper/figures/treeAnalysis',
                        help='Dir to store plots', required=False)
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed to be used', required=False)

    args = vars(parser.parse_args())

    main(args)
