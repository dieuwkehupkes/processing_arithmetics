from keras.callbacks import Callback
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# noinspection PyAttributeOutsideInit
class TrainingHistory(Callback):
    """
    Track different aspects of the network and network performance
    during training.
    """
    def __init__(self, metrics, recurrent_id,  save_every, filename, param_id=1):
        """
        :param metrics:        metrics to be monitored during training
        :param recurrent_id:   id of the recurrent layer 
        :param save_every:     store model every save_every epochs
        :param filename:       filename to write trained model to
        :param_id:             
        """
        assert (isinstance(metrics, dict) or isinstance(metrics, list))
        self.recurrent_id = recurrent_id
        self.param_id = param_id
        self.metrics = metrics
        if save_every:
            self.save_every = save_every
        else:
            self.save_every = float("inf")
        self.filename = filename
        if isinstance(metrics, list):
            self.metrics = {'output': metrics}

    def on_train_begin(self, logs={}):

        # if metrics is a dictionary, model has multiple outputs
        self.losses = dict([(output, []) for output in self.metrics])
        self.val_losses = dict([(output, []) for output in self.metrics])
        self.prediction_error = dict([(output, []) for output in self.metrics])

        self.metrics_train = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set
        self.metrics_val = dict([(output, dict([(self.metrics[output][metric], []) for metric in xrange(len(self.metrics[output]))])) for output in self.metrics])       # track metrics on training set

        self.esp = []                   # track esp recurrent layer
        self.i = 0                      # track epoch nr

    def on_epoch_end(self, epoch, logs={}):
        """
        Append tracking objects to their respective lists
        """
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

        if self.i % self.save_every == 0 and self.i != 0:
            self.write_to_file()

        self.i += 1

        # compute esp
        # recurrent_weights = self.model.layers[self.recurrent_id].get_weights()[self.param_id]
        # spec = np.max(np.absolute(np.linalg.eig(recurrent_weights)[0]))
        # self.esp.append(spec)


    def on_train_end(self, logs):
        self.write_to_file()

    def write_to_file(self):
        """
        Save model to file.
        """
        if not self.filename:
            return
        elif self.save_every == float("inf"):
            filename = self.filename
        else:
            filename = self.filename+'_'+str(self.i)

        self.model.save(filename+'.h5', overwrite=False)


class VisualiseEmbeddings(Callback):

    def __init__(self, dmap, embeddings_id):
        """
        Plot embeddings
        """
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.embeddings_id = embeddings_id
        self.dmap = dict(zip(dmap.values(), dmap.keys()))
        self.cmap = self.make_cmap(self.dmap)

    def on_train_begin(self, logs={}):
        # check if embeddings have correct dimensionality
        assert self.model.layers[self.embeddings_id].get_weights()[0].shape[1] == 2, "only 2D embddings can be visualised"

        img = []
        weights = self.model.layers[self.embeddings_id].get_weights()[0]
        for i in xrange(1, len(weights)):
            xy = tuple(weights[i])
            x, y = xy
            img += self.ax.plot(x, y, 'o')
            img.append(self.ax.annotate(self.dmap[i], xy=xy))
        plt.plot()
        self.imgs = [img]
        self.i=0

    def on_batch_end(self, batch, logs={}):
        # get snapshot of the embedding weights every 20 batches
        if self.i % 50 == 0:
            weights = self.model.layers[self.embeddings_id].get_weights()[0]
            img = []
            for i in xrange(1, len(weights)):
                xy = tuple(weights[i])
                x, y = xy
                img += self.ax.plot(x, y, 'o', color=self.cmap[i])
                img.append(self.ax.annotate(self.dmap[i], xy=xy))

            plt.plot()
            self.imgs.append(img)
        self.i+=1


    def on_train_end(self, logs={}):
        # Create animation of weight changes
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=500, blit=False, repeat_delay=3000)
        plt.show()

    def make_cmap(self, dmap):
        N = len(dmap)-4
        colorscale = plt.get_cmap('summer', N) 
        cmap = {}
        for word_id in dmap:
            try:
                cmap[word_id] = colorscale(int(dmap[word_id])+10)
            except ValueError:
                if dmap[word_id] in ['(', ')']:
                    cmap[word_id] = 'black'
                elif dmap[word_id] in ['+', '-']:
                    cmap[word_id] = 'red'
                else:
                    print("This is not supposed to happen")
        
        return cmap



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
