from keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    Track different aspects of the network and network performance
    during training.
    """
    def __init__(self, metrics, recurrent_id,  save_every, filename, param_id=1):
        assert (isinstance(metrics, dict) or isinstance(metrics, list))
        self.recurrent_id = recurrent_id
        self.param_id = param_id
        self.metrics = metrics
        if save_every:
            self.save_every = save_every
        else:
            self.save_every = float("inf")
        self.filename = filename
        if isinstance(metrics, dict):
            self.mo = True
        else:
            self.mo = False

    def on_train_begin(self, logs={}):

        # if metrics is a dictionary, model has multiple outputs
        if self.mo:
            self.on_train_begin_mo(logs)
        else:
            self.on_train_begin_1o(logs)

        self.esp = []                   # track esp recurrent layer
        self.i = 0                      # track epoch nr

    def on_train_begin_1o(self, logs):

        self.losses = []                # track training loss
        self.val_losses = []            # track validation loss
        self.prediction_error = []      # track training prediction error
        self.metrics_train = dict([(metric, []) for metric in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(metric, []) for metric in self.metrics])         # treck metrics on validation set

    def on_train_begin_mo(self, logs):
        self.losses = dict([(output, []) for output in self.metrics])
        self.val_losses = dict([(output, []) for output in self.metrics])
        self.prediction_error = dict([(output, []) for output in self.metrics])

        self.metrics_train = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set


    def on_epoch_end(self, epoch, logs={}):
        """
        Append tracking objects to their respective lists
        """
        if self.mo:
            self.on_epoch_end_mo(epoch, logs)
        else:
            self.on_epoch_end_1o(epoch, logs)

        if self.i % self.save_every == 0 and self.i != 0:
            self.write_to_file()

        self.i += 1

        # compute esp
        # recurrent_weights = self.model.layers[self.recurrent_id].get_weights()[self.param_id]
        # spec = np.max(np.absolute(np.linalg.eig(recurrent_weights)[0]))
        # self.esp.append(spec)

    def on_epoch_end_1o(self, epoch, logs):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        for metric in self.metrics:
            self.metrics_train[metric].append(logs.get(metric))
            self.metrics_val[metric].append(logs.get('val_'+metric))

    def on_epoch_end_mo(self, epoch, logs):
        for output in self.metrics:
            if len(self.metrics) == 1:
                name = ''
            else:
                name = output + '_'

            self.losses[output].append(logs.get(name+'loss'))
            self.losses[output].append(logs.get('val_'+name+'loss'))

            for metric in self.metrics[output]:
                self.metrics_train[output][metric].append(logs.get(name+metric))
                self.metrics_val[output][metric].append(logs.get('val_'+name+metric))


    def on_train_end(self, logs):
        self.write_to_file()


    def write_to_file(self):
        """
        Save model to file.
        """
        if not self.filename:
            return
        f = self.filename+str(self.i)+'.h5'
        self.model.save(f, overwrite=False)


class PlotEmbeddings(Callback):

    def __init__(self, N, dmap, embeddings_id):
        """
        Plot embeddings
        :param N:   plot embeddings every N epochs
        """
        self.N = N
        self.embeddings_id = embeddings_id
        self.dmap = dict(zip(dmap.values(), dmap.keys()))

    def on_train_begin(self, logs={}):
        # check if embeddings have correct dimensionality
        assert self.model.layers[self.embeddings_id].get_weights()[0].shape[1] == 2, "only 2D embddings can be visualised"

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
        weights = self.model.layers[self.embeddings_id].get_weights()[0]
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


class DrawWeights(Callback):

    def __init__(self, figsize, layer_id=1, param_id=0):
        self.layer_id = layer_id
        self.param_id = param_id
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def on_train_begin(self, logs={}):
        # check if weights of layer_id are accessible
        try:
            self.model.layers[self.layer_id].get_weights()[self.param_id]
        except IndexError:
            print("Weights of layer %s cannot be visualised" % self.model.layers[self.layer_id].name)
        self.imgs = []

    def on_epoch_end(self, epoch, logs={}):
        # Get a snapshot of the weight matrix every 5 batches
        if epoch % 5 == 0:
            # Access the full weight matrix
            weights = self.model.layers[self.layer_id].get_weights()[self.param_id]
            # Create the frame and add it to the animation
            img = self.ax.imshow(weights, interpolation='nearest', aspect='auto')
            plt.plot()
            self.imgs.append([img])

    def on_train_end(self, logs={}):
        # Once the training has ended, display the animation
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=500, blit=False, repeat_delay=3000)
        pcm = self.ax.get_children()[2]
        plt.colorbar(pcm, ax=self.ax)
        plt.show()
