import numpy as np
import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight("confusion_matrix", shape=(num_classes, num_classes), initializer="zeros")

    def reset_state(self):
        for state in self.variables:
            state.assign(tf.zeros(shape=state.shape))

    def update_state(self, y, y_pred):
        # convert from possibility to boolean
        # y_pred = tf.math.argmax(y_pred, axis=1)
        confusion_matrix = tf.math.confusion_matrix(y, y_pred, dtype=tf.float32, num_classes=self.num_classes)

        self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        return self.confusion_matrix

    def calculateConfusionMatrix(self, y, y_pred):
        return tf.math.confusion_matrix(y, y_pred, dtype=tf.float32, num_classes=self.num_classes)

    