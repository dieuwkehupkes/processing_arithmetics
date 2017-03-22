from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import GRU
import numpy as np

class GRU_output_gates(GRU):
    """
    Gated Recurrent Unit

    Identical from regular GRU, except that gate
    values are considered part of the output, such
    that intermediate values can be monitored.

    """
    def __init__(self, output_dim, **kwargs):
        print("Running network with adapted GRU layer")
        super(GRU_output_gates, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(GRU_output_gates, self).build(input_shape)

    def call(self, x, mask=None):
        return super(GRU_output_gates, self).call(x, mask)

    def step(self, x, states):
        """
        Step function called to compute the next state of the network
        This step function is equal to the regular GRU step function,
        except that the input
        :param x:
        :param states:
        :return:
        """

        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh

        # concatenate hidden layer activation and gate values
        all = K.concatenate([h, z, r])

        return all, [h]


    def get_output_shape_for(self, input_shape):
        return super(GRU_output_gates, self).get_output_shape_for(input_shape)
