import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gin


@gin.configurable
def rnn_model(num_rnn_neurons=256, num_rnn_layers=1,  dropout_rate=0.01, num_dense_neurons=10, num_dense_layers=1, rnn_type='lstm', ds_info=250):

    WindowSize = ds_info

    model = keras.Sequential()
    for i in range(num_rnn_layers):
        if rnn_type == 'lstm':
            model.add(layers.LSTM(num_rnn_neurons, dropout=dropout_rate,
                      return_sequences=True, input_shape=(WindowSize, 6)))
            # Output shape (None, window_size, num_neurons)
        if rnn_type == 'gru':
            model.add(layers.GRU(num_rnn_neurons, dropout=dropout_rate,
                      return_sequences=True, input_shape=(WindowSize, 6)))

    for i in range(num_dense_layers):
        model.add(layers.Dense(num_dense_neurons, activation='relu'))
        # Output shape (None, window_size, num_neurons)

    model.add(layers.Dense(12, activation='softmax'))
    # Output shape (None, window_size, num_neurons)

    return model


if __name__ == '__main__':
    model = rnn_model(128, 2, 0.5, 10, 2, 'gru')
    model.summary()
