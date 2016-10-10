import pickle
import argparse
import data
import numpy as np


import plotTools as pt

def plotEmbs(theta):
  wm = theta[('word',)]
  mbits = np.split(theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')], 3, axis=1)
  fig,ax = pt.newFig()
  for w, v in wm.iteritems():
      if w[-1].isdigit():
          pt.scatterAndAnnotate(ax,point=v,label=w,color='green')
          pt.scatterAndAnnotate(ax, point=mbits[0].dot(v), label=w+'L', color='red')
          pt.scatterAndAnnotate(ax, point=mbits[2].dot(v), label=w+'R', color='blue')
      else: print w
  pt.title('Effect of composition as left or right child on embeddings')
  pt.show()

  # for digit in ds:
  #   rep = theta[('word',)][str(digit)]
  #   plt.scatter(rep[0],rep[1],color='green')
  #   plt.annotate(digit, xy=(rep[0],rep[1]))
  # for operator in ops:
  #   rep = theta[('word',)][operator]
  #   plt.scatter(rep[0],rep[1],color='red')
  #   plt.annotate(operator, xy=(rep[0],rep[1]))


def plotMatrices(theta):
    for name, mat in theta.iteritems():
        if 'M'in name:
            pt.showmat(mat, cb=True)
            pt.title(name)
            pt.show()

def comparison(theta):
    try: left, right = np.split(theta[('comparison', 'M')], 2, axis=1)
    except:
        print 'no comparison matrix in theta'
        return

    mtb = data.arithmetics.training_treebank(args['seed'], languages={'L9': 25}, )
    rnnTB = data.RNNTB(mtb.examples)
    colors = pt.colorscale(120)
    for i in range(6):
        for j in range(i+1,6):

            fig, ax = pt.newFig()


            for rnn, tar in rnnTB.getExamples():
                rep = rnn.activate(theta)
                leftie = left.dot(rep)
                rightie = right.dot(rep)
                #print leftie.shape, rightie.shape
                #pt.scatterAndAnnotate(ax, rep, color=colors(tar+60), label=tar, size=8, txtcolor='black')
                pt.scatterAndAnnotate(ax, (leftie[i],leftie[j]), color=colors(tar+60), label=str(tar)+'L', size=8, txtcolor='black')
                pt.scatterAndAnnotate(ax, (rightie[i],rightie[j]), color=colors(tar+60), label=str(tar)+'R', size=8, txtcolor='black')

            pt.title('Effect of comparison layer on L9 expressions, dims '+str(i)+'-'+str(j))
            pt.show()


def plotAnswers(theta):
    mtb = data.arithmetics.training_treebank(args['seed'], languages={'L9':2500},)
    rnnTB = data.RNNTB(mtb.examples)
    fig,ax = pt.newFig()
    colors = pt.colorscale(120)
    tolabel=range(-60,61,5)
    tolabel=tolabel+tolabel[:]+tolabel[:]
    for rnn, tar in rnnTB.getExamples():
        rep = rnn.activate(theta)
        if tar in tolabel:
            pt.scatterAndAnnotate(ax, rep, color=colors(tar+60), label=tar, size=8, txtcolor='black')
            del tolabel[tolabel.index(tar)]
        else: pt.scatterAndAnnotate(ax, rep, color=colors(int(tar)+60), label='', size=5, txtcolor='black')
    pt.title('Root representations of TreeRNN L9 expressions')
    pt.show()

def main(args):
    with open(args['theta'],'rb') as f:
        theta = pickle.load(f)
    comparison(theta)
    plotMatrices(theta)
    plotAnswers(theta)
    plotEmbs(theta)




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train classifier')
  parser.add_argument('-t','--theta', type=str, help='File with pickled theta', required=True)
  parser.add_argument('-o', '--outDir', type=str, help='Dir to store plots', required=True)
  parser.add_argument('-s', '--seed', type=int, default = 42, help='Random seed to be used', required=False)

  args = vars(parser.parse_args())

  main(args)


