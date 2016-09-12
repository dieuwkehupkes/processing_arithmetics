from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

class MonitorUpdates(Callback):
    """
    Monitor the size of the updates for the embeddings over time
    time
    """

    def on_train_begin(self, logs={}):
        self.prev_weights = self.model.layers[1].get_weights()[0]
        self.cur_weights = self.model.layers[1].get_weights()[0]
        self.emb_size = len(self.prev_weights)
        self.batch_updates = np.zeros(self.emb_size).reshape(self.emb_size, 1)

    def on_epoch_end(self, epoch, logs={}):
        self.cur_weights = self.model.layers[1].get_weights()[0]
        updates = np.absolute(self.cur_weights - self.prev_weights)
        updates_av = np.mean(updates, axis=1).reshape(self.emb_size, 1)
        self.prev_weights = self.cur_weights
        self.batch_updates = np.append(self.batch_updates, updates_av, axis=1)

        # plot dev updates until now
        if epoch % 20 == 0 and epoch!= 0:
            self.plot()

    def plot(self):
        plt.plot(self.batch_updates[0], linewidth=3.0, label='updates weight 0')
        for update_sequence in self.batch_updates:
            plt.plot(update_sequence)
        plt.legend()
        plt.show()



