import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as coloring
from matplotlib.patches import Ellipse, Polygon
import pickle
import core.myRNN as RNN
import sys
from collections import defaultdict
from core.activation import activate
import os
from nltk import Tree

#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')


def main():
  global save, d3, saveDir, model, name
  d3=False
#  model='9876878'
  #model='9948528'
  model = sys.argv[1]
  save = True
  saveDir= '../mathAnalysis/plots/'+model+'/'
  if not os.path.exists(saveDir): os.makedirs(saveDir)
  try: k = sys.argv[2]
  except: k= 'FFF1' 
  filename = '../mathAnalysis/models/'+model+'/d2-a0.2-b32-l0.0002-fe'+k[0]+'-fc'+k[1]+'-nc'+k[2]+'-e'+k[3]+'/plainTrain.theta.pik'
  name='d2'+k
  initialize(filename, False)

  d2Plots()

def getpca():
  M= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
  mbits= np.split(M,3,axis=1)

  lefties = [mbits[0].dot(theta[('word',)][w]) for w in digits]
  righties = [mbits[2].dot(theta[('word',)][w]) for w in digits]
  return PCA(np.array(lefties+righties), False)

def initialize(filename, toPCA=False):
  global theta, digits,operators
  with open(filename,'rb') as f:
    theta = pickle.load(f)

  voc = theta[('word',)].keys()
  digits = []
  for w in voc:
    try:
      d=   int(w)
      digits.append(d)
    except: continue
  operators = [w for w in voc if w in ['plus','minus','times','div','modulo']]

  global pca
  if toPCA:
    pca=getpca()
    print 'PCA'
  else: pca = None


def d2Plots():

  tb = math.mathTreebank(operators, digits, 10, range(4,8))

  plotEmbs(digits,operators)
  plotAnswers(tb)
  plotAnswersCM(tb)#,'plus')
  plotProjectedEmbs()
  plotExamples(5)
  plotAnExample()
  plotMPart()


def plotMPart():
  fig,ax = newFig()
  M= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
  mbits= np.split(M,3,axis=1)
  stuff = []
  if pca is None:
    print 'pca is None'
    for i in np.arange(-1.2,1.2,0.2):
      for j in np.arange(-1.2,1.2,0.2):
        loc=np.array([i,j])
        rep = mbits[0].dot(loc)
        stuff+=scatterAndAnnotate(plt, rep, color='red', name='' )
        stuff += drawArrow(plt,loc,rep,'',style='solid')
  else:
    for p in range(100):
      loc = np.random.random_sample(theta[('word',)][digits[0]].shape)-0.5#*2.4-1
      rep =mbits[0].dot(loc)
#      print repI
      stuff+=scatterAndAnnotate(plt, rep, color='pink', name='' )
      stuff += drawArrow(plt,pca.project(loc),rep,'',style='solid')


  if save: plt.savefig(saveDir+name+'compositionLeft.png')
  else: plt.show()

  for i in stuff: i.remove()
  if pca is None:
    print 'pca is None'
    for i in np.arange(-1.2,1.2,0.2):
      for j in np.arange(-1.2,1.2,0.2):
   #     print 'i,j:',i, j
        loc=np.array([i,j])
        rep = mbits[2].dot(loc)
     #   print 'loc:',loc,'rep:', rep
        stuff+=scatterAndAnnotate(plt, rep, color='green', name='' )
        #plt.scatter(rep[0],rep[1],color='green')
        stuff += drawArrow(plt,loc,rep,'',style='solid')
  else:
    for p in range(100):
      loc = np.random.random_sample(theta[('word',)][digits[0]].shape)-0.5#*2.4-1
      rep = mbits[2].dot(loc)
#      stuff.append(plt.scatter(rep[0],rep[1],color='green'))
      stuff+=scatterAndAnnotate(plt, rep, color='green', name='' )
      stuff += drawArrow(plt,pca.project(loc),rep,'',style='solid')


  if save: plt.savefig(saveDir+name+'compositionRight.png')
  else: plt.show()



def plotEmbs(ds,ops,fig=None, ax = None):
  if fig is None: fig,ax = newFig()
  for digit in ds:
    rep = theta[('word',)][str(digit)]
    plt.scatter(rep[0],rep[1],color='green')
    plt.annotate(digit, xy=(rep[0],rep[1]))
  for operator in ops:
    rep = theta[('word',)][operator]
    plt.scatter(rep[0],rep[1],color='red')
    plt.annotate(operator, xy=(rep[0],rep[1]))

def plotAnswersCM(tb,op=None):
  fig,ax = newFig()
  toPlot=defaultdict(list)
  colormap=plt.get_cmap('cool',30)
  colormap1=plt.get_cmap('winter',30)
  colormap2=plt.get_cmap('summer',30)
  colormap3=plt.get_cmap('autumn',30)
  colormap4=plt.get_cmap('spring',30)

  M= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
  mbits= np.split(M,3,axis=1)

  stuff = []
  for t, a in tb.examples[:500]:
      if a<-15 or a>15: continue
#      print t, a
      nw = RNN.RNN(t)
      if op is not None:
        try: assert nw.root.inputs[1].key==op
        except: continue
      nw.activate(theta)
      rep=nw.root.a
      lop=str(nw.root.inputs[1])
#      print lop
      if lop=='minus':
        lop=''#'-'
        scatterAndAnnotate(ax, rep, color=colormap2(a+15), name=lop,small=True)
      elif lop=='plus':
        lop=''#'+'
        scatterAndAnnotate(ax, rep, color=colormap1(a+15), name=lop,small=True)
      elif lop in ['+','-']:
        scatterAndAnnotate(ax, rep, color=colormap1(a + 15), name=lop, small=True)
      else: print lop

#      plt.scatter(rep[0],rep[1],color=colormap(a+50))
      left = mbits[0].dot(rep)
      stuff+= scatterAndAnnotate(ax, left, color=colormap3(a+15), name=lop,small=True)
#      plt.scatter(left[0],left[1],color=colormap2(a+50))
      right = mbits[2].dot(rep)
      stuff+= scatterAndAnnotate(ax, right, color=colormap4(a+15), name=lop,small=True)
#      plt.scatter(right[0],right[1],color=colormap2(a+50))

  if save: plt.savefig(saveDir+name+'answersCMNew01.png')
  else: plt.show()

  for p in stuff: p.remove()
  if save: plt.savefig(saveDir+name+'answersCMNew02.png')
  else: plt.show()



def plotAnswers(tb):
  fig,ax = newFig()
  for t, a in tb.examples[:20]:
#      print a
      nw = RNN.RNN(t)
      nw.activate(theta)
      leaves = nw.leaves()
      result = nw.root.a
      colors = ['r','b', 'r','g']
  #    for i in range(len(leaves)):
  #      plt.scatter(leaves[i].a[0],leaves[i].a[1],color='blue')
  #      plt.annotate(leaves[i].key, xy=(leaves[i].a[0],leaves[i].a[1]))
      plt.scatter(result[0],result[1],color='blue')
      plt.annotate(a, xy=(result[0],result[1]))

def plotExamples(n=1):
  fig,ax = newFig()
  tb = math.mathTreebank(operators, digits, 10, range(2,3))
  colors=iter(['red','blue','green','yellow'])
  ops=dict()
  plotEmbs(digits,[],fig,ax)
  for operator in operators:
    rep = theta[('word',)][operator]
    color = colors.next()
    plt.scatter(rep[0],rep[1],color=color)
    plt.annotate(operator, xy=(rep[0],rep[1]))
    ops[operator]=color
  color = colors.next()
  #fig = plt.figure()
  for t, a in tb.examples[:n]:
#      print a
      nw = RNN.RNN(t)
      nw.activate(theta)
      leaves = nw.leaves()
      result = nw.root.a
      operator=leaves[1].key


      ax.add_patch(Polygon([leaves[0].a,leaves[2].a, result], closed=True,
                      fill=False, color=ops[operator]))

#       for i in range(len(leaves)):
#         plt.scatter(leaves[i].a[0],leaves[i].a[1],color='red')
#         plt.annotate(leaves[i].key, xy=(leaves[i].a[0],leaves[i].a[1]))
      plt.scatter(result[0],result[1],color=ops[operator])
      plt.annotate(a, xy=(result[0],result[1]))

#      plt.scatter(leaves[1].a[0],leaves[1].a[1],color='blue')
#      plt.annotate(operator, xy=(leaves[1].a[0],leaves[1].a[1]))


def plotAnExample(n=1):
  fig,ax = newFig(True)

  M= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
  mbits= np.split(M,3,axis=1)
  B= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'B')]
 # tb = math.mathTreebank(operators, digits, 10, range(2,3))
  t= Tree('-',[Tree('digit',['2']),Tree('operator',['minus']),Tree('digit',['3'])])
  a=10
#  for t,a in tb.getExamples(1):

  colors = ['green', 'red', 'blue']
  if True:
    nw = RNN.RNN(t)
    nw.activate(theta)
    leaves = nw.leaves()
    parts = [mbits[i].dot(leaves[i].a) for i in range(3)]

#    colors=['green','red','green']
#    colors2=['yellow','orange','blue']
#    colors3=['red','orange','magenta']
    embs = []
#    projs = []

    # create figure with embeddings and projections
    for i in range(3):
      rep = leaves[i].a
      proj =  parts[i]
#      embs+=scatterAndAnnotate(ax,rep , colors[i], leaves[i].key)
      embs+=scatterAndAnnotate(ax,rep ,'green', leaves[i].key)
      embs += drawArrow(ax, rep, proj,'',head=0.1)#, 'project-' + str(i), head=0.1)
#      embs += scatterAndAnnotate(ax, proj, 'red', leaves[i].key+'\'')

    #      projs += scatterAndAnnotate(ax,proj, colors2[i], leaves[i].key+'\'')

    if save: plt.savefig(saveDir + name + 'example05project.png')
    for i in embs: i.remove()

    #if save: plt.savefig(saveDir+name+'example01projections.png')
#    else: plt.show()

    # get rid of the embeddings
#    for i in embs: i.remove()

    # todel = []
    # for i in range(3):
    #   proj =  parts[i]
    #   todel += scatterAndAnnotate(ax,proj, colors3[i], leaves[i].key+'\'')
    #   todel += drawArrow(ax,(0,0),proj,'',head=0.1)


      #if save: plt.savefig(saveDir+name+'example01projectionsOnly.png')
    #else: plt.show()
#    for i in todel: i.remove()

#    for i in projs: i.remove()
    arrows=[]
    oldloc = np.zeros_like(B) #(0,0)
    newloc = oldloc+B
    arrows+=drawArrow(ax,oldloc,newloc,r'bias',head=0.1)
    for i in [0,2,1]:
      oldloc= newloc
      newloc = oldloc+parts[i]
      try: arrows+=drawArrow(ax,oldloc,newloc,leaves[i].key+'\'',head=0.1)
      except: continue
    result=newloc
    scatterAndAnnotate(ax, result, 'blue', '')
    newloc = oldloc+mbits[1].dot(theta[('word',)]['minus'])
    try: arrows+=drawArrow(ax,oldloc,newloc,r'minus\'', 'dotted',head=0.1)
    except: None
#    projs+=scatterAndAnnotate(ax, B, 'red', 'bias')
    if save: plt.savefig(saveDir+name+'example06sum.png')
    else: plt.show()


    for i in arrows: i.remove()
#    sm=sum(parts)+B
    asm = activate(result,'tanh')[0]
    scatterAndAnnotate(ax, asm, 'blue', a)
    drawArrow(ax,result,asm,'tanh',head=0.1)
    drawArrow(ax,result,asm,'tanh',head=0.1)
    asmMinus = activate(newloc,'tanh')[0]
    drawArrow(ax,newloc,asmMinus,'tanh','dotted',head=0.1)
    if save: plt.savefig(saveDir+name+'example07squash.png')
    else: plt.show()


def plotProjectedEmbs():
  fig,ax = newFig()
  M= theta[('composition', '#X#', '(#X#, #X#, #X#)', 'I', 'M')]
  mbits= np.split(M,3,axis=1)
  todel = []
  for digit in digits:
    rep = theta[('word',)][str(digit)]
    scatterAndAnnotate(ax,rep,'green',str(digit))
    todel+= scatterAndAnnotate(ax,mbits[0].dot(rep),'red',str(digit))
    todel +=scatterAndAnnotate(ax,mbits[2].dot(rep),'magenta',str(digit))
#  for p in todel: p.remove()
#  for operator in operators:
#    rep = theta[('word',)][str(operator)]
#    scatterAndAnnotate(ax,rep,'red',str(operator))
#    scatterAndAnnotate(ax,mbits[1].dot(rep),'orange',str(operator))

  if save: plt.savefig(saveDir+name+'projectedEmbs.png')
  else: plt.show()


def scatterAndAnnotate(ax, point, color, name,small=False,apca=None):
  if small:
    size=5
    txtcolor='gray'
  else:
    size =12
    txtcolor='black'
  if d3:   sc = ax.scatter(float(point[0]),float(point[1]),float(point[2]),color=color,s=25)
  else: sc = ax.scatter(float(point[0]),float(point[1]),color=color,s=25)
  an = ax.annotate(name, xy=(float(point[0]),float(point[1])),size=size, color=txtcolor)
  return [sc,an]


def drawArrow(ax,fr,to,name,style='solid', head=0.05):
  if pca is not None:
    fr = pca.project(fr)
    to = pca.project(to)
    
  if fr[0] == to[0] and fr[1]==to[1]:
    print 'Not drawing an arrow from and to the same point! Exit function drawArrow'
    raise Exception()

  dx = to[0]-fr[0]
  dy = to[1]-fr[1]
  try: rotation = int(round(np.rad2deg(np.arctan(dy/dx))))
  except: rotation = 1
  #print rotation
  ar = ax.arrow(fr[0],fr[1],dx,dy,width=0.0001,ls=style,head_width=head, color='grey', length_includes_head=True, label='project')
  if style == 'solid':
    an = ax.text(fr[0]+.5*dx, fr[1]+.5*dy, name, ha="center", va="center", rotation=rotation,size=10)#,bbox=dict(facecolor='white', edgecolor='none', fill=True, alpha=0.8))
  else:
    an = ax.text(fr[0]+.5*dx, fr[1]+.5*dy, name, color='grey',ha="center", va="center", rotation=rotation,size=8)#,bbox=dict(facecolor='white', edgecolor='none', fill=True, alpha=0.8))
  return [ar, an]

def newFig(ticks=False):
  plt.close()
  if d3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  else: fig, ax = plt.subplots()

  ax.spines['left'].set_position('zero')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_position('zero')
  ax.spines['right'].set_position('zero')

  if ticks:
    plt.xticks([-5,-4,-3,-2,-1,1,2,3,4,5])
    plt.yticks([-5,-4,-3,-2,-1,1,2,3,4,5])
    ax.set_ylim([-7,7])
    ax.set_xlim([-7,7])

  return fig, ax

if __name__ == "__main__": main()
