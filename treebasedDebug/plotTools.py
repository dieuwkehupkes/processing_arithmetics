import matplotlib.pyplot as plt

def colorscale(n):
  return plt.get_cmap('gist_rainbow', n)

def scatterAndAnnotate(ax, point, color, label,size=12,txtcolor='black'):
  if len(point)>2: print 'Only plotting first two dims'
  sc = ax.scatter(float(point[0]),float(point[1]),color=color,s=25)
  an = ax.annotate(label, xy=(float(point[0]),float(point[1])),size=size, color=txtcolor)
  return [sc,an]

def title(title):
    plt.title(title)

def showmat(M,ax=None, cb=False):
  if ax is None:
    r= plt.matshow(M,vmin=-1,vmax=1,cmap='bwr')
    cb = plt.colorbar(location='right', ticks=[], fraction=0.1, pad=0.05)
    cb.ax.text(0.5, -0.01, '-1', transform=cb.ax.transAxes, va='top', ha='center')
    cb.ax.text(0.5, 0.5, '0', transform=cb.ax.transAxes, va='center', ha='center')
    cb.ax.text(0.5, 1.0, '1', transform=cb.ax.transAxes, va='bottom', ha='center')
  else: return ax.matshow(M,vmin=-1,vmax=1,cmap='bwr')


def newFig():
  plt.close()
  fig, ax = plt.subplots()

  # ax.spines['left'].set_position('zero')
  # ax.spines['bottom'].set_position('zero')
  # ax.spines['top'].set_position('zero')
  # ax.spines['right'].set_position('zero')

  return fig, ax

def improveTicks(ax,scale=1):
  ax.set_xticks([scale*v for v in ax.get_xticks() if v!=0])
  ax.set_yticks([scale*v for v in ax.get_yticks() if v!=0])
  ax.spines['left'].set_position('zero')
  ax.spines['bottom'].set_position('zero')

  # turn off the other spine/ticks
  ax.spines['right'].set_color('none')
  ax.yaxis.tick_left()
  ax.spines['top'].set_color('none')
  ax.xaxis.tick_bottom()


def show():
    plt.show()