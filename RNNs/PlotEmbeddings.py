from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PlotEmbeddings(Callback):

    def __init__(self, N, dmap):
        """
        Plot embeddings
        :param N:   plot embeddings every N epochs
        """
        self.N = N
        self.dmap = dict(zip(dmap.values(), dmap.keys()))

    def on_train_begin(self, logs={}):
        # check if embeddings have correct dimensionality
        assert self.model.layers[1].get_weights()[0].shape[1] == 2, "only 2D embddings can be visualised"

    def on_epoch_end(self, epoch, logs={}):
        # Get a snapshot of the weight matrix every 5 batches
        if epoch % self.N == 0 and epoch != 0:
            # plot embeddings
            self.plot()

    def on_train_end(self, logs={}):
        # Plot weights last time after training ended
        self.plot()

    def plot(self):
        """
        Plot weights
        """
        weights = self.model.layers[1].get_weights()[0]
        # find limits
        xmin, ymin = 1.1 * weights.min(axis=0)
        xmax, ymax = 1.1 * weights.max(axis=0)
        plt.clf()
        for i in xrange(1, len(weights)):
            xy = tuple(weights[i])
            x, y = xy
            plt.plot(x, y, 'o')
            plt.annotate(self.dmap[i], xy=xy)
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
        plt.show()
