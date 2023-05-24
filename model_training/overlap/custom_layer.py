import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K

class WrapperLayer(tf.keras.layers.layer):
    def __init__(self, output_dim, **kwargs):
        self.units = output_dim
        super(WrapperLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(WrapperLayer, self).build(input_shape)

    def call(self, x):
        y = K.dot(x, self.kernel)
        return y

    def call(self, inputs):
        loc = self.loc(inputs)
        scale_vec = self.scale(inputs)
        theta = self.rotation(inputs)
        scale_diag = tf.linalg.diag(scale_vec)
        rotation_mat = tf.reshape(tf.stack([tf.math.cos(theta), -tf.math.sin(theta), tf.math.sin(theta), tf.math.cos(theta)], axis=1), (tf.shape(theta)[0], 2, 2))
        scale_mat = tf.matmul(rotation_mat, scale_diag)
        params = [loc, scale_mat]
        density = tfp.layers.DistributionLambda(
            make_distribution_fn = lambda params: tfd.MultivariateNormalLinearOperator(loc=params[0], scale=tf.linalg.LinearOperatorFullMatrix(params[1])),
        )(params)
        return density

    def get_config(self):
        config = dict()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)