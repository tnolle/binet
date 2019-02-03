# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import tensorflow as tf
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Layer


class Attention(Layer):
    """Attention layer implementation based on the code from Baziotis, Christos on
    Gist (https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2) and on
    the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification".

    The difference between these implementations are:
    - Compatibility with Keras 2.2;
    - Code annotations;
    - Easy way to retrieve attention weights;
    - Tested with TensorFlow backend only!
    """

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product between a matrix and a vector x*u.
        The shapes are arranged as (batch_size, timesteps, features) \
        * (features,) = (batch_size, timesteps)
        Args:
            x: Matrix with shape (batch_size, timesteps, features)
            kernel: Vector with shape (features,)
        Returns:
            W = x*kernel with shape (batch_size, timesteps)
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)

    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, return_sequences=False, concat_len=1, **kwargs):
        """If return_coefficients is True, this layer produces two outputs,
        the first is the weighted hidden tensor for the input sequence with
        shape (batch_size, n_features) and the second the attention weights
        with shape (batch_size, seq_len, 1). This layer supports masking."""

        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.return_sequences = return_sequences
        self.init = initializers.get('glorot_uniform')
        self.concat_len = concat_len
        self.masks = None

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        # The attention vector/matrix is equals to the RNN hidden dimension
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        # Setup time masks
        self.masks = K.ones((input_shape[1] / self.concat_len, input_shape[1] / self.concat_len))
        self.masks = tf.matrix_band_part(self.masks, -1, 0)

        if self.concat_len > 1:
            self.masks = K.tile(self.masks, [1, self.concat_len])

        super(Attention, self).build(input_shape)

    @staticmethod
    def masked_softmax(alpha, mask):
        """Masks alpha and then performs softmax, as Keras's softmax doesn't support masking."""
        alpha = K.exp(alpha)
        if mask is not None:
            alpha *= K.cast(mask, K.floatx())

        partition = K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return alpha / partition

    def call(self, x, mask=None):
        v_att = K.dot(x, self.W)
        if self.bias:
            v_att += self.b
        v_att = K.tanh(v_att)

        mask = K.cast(mask, K.floatx())
        mask = K.expand_dims(mask, axis=1)

        masks = self.masks * mask
        masks = K.permute_dimensions(masks, [1, 0, 2])

        alpha = self.dot_product(v_att, self.u)  # > (batch, seq_len)

        alphas = K.map_fn(lambda mask: K.expand_dims(self.masked_softmax(alpha, mask)), masks)
        alphas = K.permute_dimensions(alphas, [1, 0, 2, 3])

        x = K.expand_dims(x, axis=1)
        attended_x = x * alphas

        alphas = K.squeeze(alphas, axis=3)

        cs = K.sum(attended_x, axis=2)

        if self.return_sequences:
            alpha = alphas
            c = cs
        else:
            alpha = tf.gather(alphas, -1, axis=1)
            c = tf.gather(cs, -1, axis=1)

        if self.return_coefficients:
            return [c, alpha]
        else:
            return c

    def compute_output_shape(self, input_shape):
        """The attention mechanism computes a weighted average between all
        hidden vectors generated by the previous sequential layer, hence the
        input is expected to be (batch_size, seq_len, amount_features) and
        after averaging each feature vector, the output it (batch_size, seq_len)."""
        if self.return_sequences:
            output_shape = (input_shape[0], int(input_shape[1] / self.concat_len), input_shape[2])
            coefficients_shape = (input_shape[0], int(input_shape[1] / self.concat_len), input_shape[1])
        else:
            output_shape = (input_shape[0], input_shape[-1])
            coefficients_shape = (input_shape[0], input_shape[-1])

        if self.return_coefficients:
            return [output_shape, coefficients_shape]
        else:
            return output_shape

    def compute_mask(self, x, input_mask=None):
        """This layer produces a single attended vector from a list of hidden vectors,
        hence it can't be masked as this means masking a single vector.
        """
        return None

    def get_config(self):
        config = {
            'return_coefficients': self.return_coefficients,
            'return_sequences': self.return_sequences,
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'W_constraint': constraints.serialize(self.W_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            'bias': self.bias,
            'concat_len': self.concat_len
        }

        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
