import tensorflow as tf


class IED(tf.keras.metrics.Metric):
    def __init__(self, k, n_items, name='IED', **kwargs):
        super(IED, self).__init__(name=name, **kwargs)
        self.IED = 0.
        self.k = k
        self.n_items = n_items

    def exposure(self, y_true, y_pred, sample_weight=None):
        position = tf.cast(tf.argsort(tf.argsort(y_pred, direction='DESCENDING')), tf.float32) + 1.
        mask = tf.cast(position <= self.k, tf.float32)
        bias = 1. / tf.math.log(1. + position)
        return tf.reduce_mean(bias * mask, axis=0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        i_exposure = self.exposure(y_true, y_pred, sample_weight)
        self.IED = tf.reduce_sum(tf.abs(i_exposure[:, None] - i_exposure[None, :])) / (2. * tf.cast(self.n_items, tf.float32) * tf.reduce_sum(i_exposure))

    def result(self):
        return self.IED