from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
            print "Weights of layer %s cannot be visualised" % self.model.layers[self.layer_id].name
        self.imgs = []

    def on_batch_end(self, batch, logs={}):
        # Get a snapshot of the weight matrix every 5 batches
        if batch % 5 == 0:
            # Access the full weight matrix
            weights = self.model.layers[self.layer_id].get_weights()[self.param_id]
            # Create the frame and add it to the animation
            img = self.ax.imshow(weights, interpolation='nearest', aspect='auto')
            plt.plot()
            self.imgs.append([img])

    def on_train_end(self, logs={}):
        # Once the training has ended, display the animation
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False, repeat_delay=3000)
        plt.show()