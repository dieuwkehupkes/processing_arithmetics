import matplotlib.pyplot as plt
import os


def colorscale(n, color=None):
    if color is None:
        return plt.get_cmap('gist_rainbow', n)
    elif color == 'red':
        return plt.get_cmap('autumn', n)
    elif color == 'blue':
        return plt.get_cmap('coolwarm', n)
    elif color == 'green':
        return plt.get_cmap('summer', n)
    elif color == 'purple':
        return plt.get_cmap('spring', n)


def scatter_and_annotate(ax, point, color, label, size=12, txtcolor='black', pos=None, alpha = 1):
    if len(point) > 2: print 'Only plotting first two dims'
    sc = ax.scatter(float(point[0]), float(point[1]), color=color, s=25, zorder=5, alpha = alpha)

    xy = [float(point[0]), float(point[1])]
    ha = 'left'
    va = 'center'
    if pos == 'right':
        xy[0] += 0.06
    elif pos == 'left':
        xy[0] -= 0.06
        ha = 'right'
    elif pos == 'top':
        xy[1] += 0.06
    elif pos == 'bot':
        xy[0] -= 0.2
        xy[1] -= 0.15
    if label != '':
        an = ax.annotate(label, xy=xy, size=size, color=txtcolor, zorder=10, ha=ha, va=va)
    else:
        an = None
    return [sc, an]


def title(title):
    plt.title(title)


def showmat(M, ax=None, cb=False):
    if ax is None:
        r = plt.matshow(M, vmin=-1, vmax=1, cmap='bwr')
        cb = plt.colorbar(location='right', ticks=[], fraction=0.1, pad=0.05)
        cb.ax.text(0.5, -0.01, '-1', transform=cb.ax.transAxes, va='top', ha='center')
        cb.ax.text(0.5, 0.5, '0', transform=cb.ax.transAxes, va='center', ha='center')
        cb.ax.text(0.5, 1.0, '1', transform=cb.ax.transAxes, va='bottom', ha='center')
    else:
        return ax.matshow(M, vmin=-1, vmax=1, cmap='bwr')


def draw_arrow(ax, fr, to, label, style='solid', color='black', head=0.05, txtsize=15, txtcolor='black', alpha=1):
    if fr[0] == to[0] and fr[1] == to[1]:
        raise Exception('Not drawing an arrow from and to the same point! Exit function draw_arrow')

    dx = to[0] - fr[0]
    dy = to[1] - fr[1]
    ar = ax.arrow(fr[0], fr[1], dx, dy, width=0.0001, ls=style, head_width=head, color=color, length_includes_head=True,
                  label=label, zorder=0, alpha=alpha)

    
    try: rotation = int(round(np.rad2deg(np.arctan(dy / dx))))
    except: rotation = 1
    anLoc = (fr[0]+rotation*0.2*dx, fr[1]+rotation*0.2*dy+0.1)

    an = ax.text(anLoc[0], anLoc[1], label, color=txtcolor, ha="center", va="center", rotation=1,
                 size=txtsize, zorder=1)  # ,bbox=dict(facecolor='white', edgecolor='none', fill=True, alpha=0.8))
    return [ar, an]


def new_fig(limits=None):
    plt.close()
    fig, ax = plt.subplots()
    if limits:
        ax.set_ylim(limits[1])
        ax.set_xlim(limits[0])

    # ax.spines['left'].set_position('zero')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['top'].set_position('zero')
    # ax.spines['right'].set_position('zero')

    return fig, ax


def put_origin(ax):
    y = ax.get_ylim()
    ax.set_ylim(min(y[0], -1.2), y[1])

    x = ax.get_xlim()
    ax.set_xlim(min(x[0], -1.2), x[1])

    # print x,y


def minimal_ticks(ax):
    ax.set_xticks([-1.0, 1.0])
    ax.set_yticks([-1.0, 1.0])
    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('grey')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')


def improve_ticks(ax, scale=1):
    horizontal = 0. in ax.get_yticks()
    vertical = 0. in ax.get_xticks()
    if vertical: ax.spines['left'].set_position('zero')
    if horizontal: ax.spines['bottom'].set_position('zero')
    if horizontal and vertical:
        ax.set_xticks([scale * v for v in ax.get_xticks() if v != 0])
        ax.set_yticks([scale * v for v in ax.get_yticks() if v != 0])

    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    # turn off the other spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()


def equal_aspects(ax):
    ax.axis('equal')

def show(ax, improve=True):
    if improve: improve_ticks(ax)
    plt.show()


def save_figure(ax, improve, outdir, name):
    if improve: improve_ticks(ax)
    plt.savefig(os.path.join(outdir, name))
