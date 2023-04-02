import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from input_pipeline.TFR import train_record_file, test_record_file, val_record_file, create_tfrecords
from input_pipeline.preprocessing import preprocess


@gin.configurable
def load(existed_tfrecords, name, data_dir, window_size, shift_window_size):

    # create tfrecord files if they don't exist
    create_tfrecords(existed_tfrecords, data_dir)

    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")

        # loading tfrecord file
       
        ds_train = tf.data.TFRecordDataset(train_record_file)
        ds_val = tf.data.TFRecordDataset(val_record_file)
        ds_test = tf.data.TFRecordDataset(test_record_file)

        def parse_record(record):
            name_to_features = {
                'data': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),

            }

            return tf.io.parse_single_example(record, name_to_features)

        def decode_record(record):
            data = record['data']
            data = tf.io.parse_tensor(data, out_type=tf.float64)
            label = record['label']

            return data, label

        # decoding and windowing the training data
        ds_train = ds_train.map(parse_record).map(decode_record)
        ds_train = ds_train.window(
            size=window_size, shift=shift_window_size, drop_remainder=True)

        ds_train = ds_train.flat_map(lambda data, label: tf.data.Dataset.zip((data, label))).batch(window_size,
                                                                                                   drop_remainder=True)

        # decoding and windowing the validation data
        ds_val = ds_val.map(parse_record).map(decode_record)
        ds_val = ds_val.window(
            size=window_size, shift=shift_window_size, drop_remainder=True)
        ds_val = ds_val.flat_map(lambda data, label: tf.data.Dataset.zip(
            (data, label))).batch(window_size, drop_remainder=True)

        # decoding and windowing the test data
        ds_test = ds_test.map(parse_record).map(decode_record)
        ds_test = ds_test.window(
            size=window_size, shift=shift_window_size, drop_remainder=True)
        ds_test = ds_test.flat_map(lambda data, label: tf.data.Dataset.zip((data, label))).batch(window_size,
                                                                                                 drop_remainder=True)

        ds_info = window_size

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    
    # Prepare training dataset
    ds_train = ds_train.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
