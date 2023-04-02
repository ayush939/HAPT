from input_pipeline.data_extraction import extract_data
import tensorflow as tf


train_record_file = "train.tfrecords"
val_record_file = "val.tfrecords"
test_record_file = "test.tfrecords"

# generates tfrecord files for training, test and validation data 
def create_tfrecords(existed_files=False, path="/home/data/HAPT_dataset"):

    path = path + '/RawData/'

    if not existed_files:

        train_x, train_y = extract_data(path, start=1, end=21)
        val_x, val_y = extract_data(path, start=28, end=30)
        test_x, test_y = extract_data(path, start=22, end=27)
        print("Extracted the data from raw files...")

        def _bytes_feature(value):
            # return a bytes_list form a string / byte
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def data_example(data, label):
            feature = {
                "data": _bytes_feature(data),
                "label": _int64_feature(int(label))
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))

        with tf.io.TFRecordWriter(train_record_file) as writer:
            for i in range(len(train_x.index)):
                data = train_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = train_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())
        print("Created TFRecord for train data")

        with tf.io.TFRecordWriter(val_record_file) as writer:
            for i in range(len(val_x.index)):
                data = val_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = val_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())
        print("Created TFRecord for val data")

        with tf.io.TFRecordWriter(test_record_file) as writer:
            for i in range(len(test_x.index)):
                data = test_x.iloc[i].values
                data = tf.io.serialize_tensor(data)
                # print(data)
                label = test_y.iloc[i].values
                label = tf.convert_to_tensor(label, dtype=tf.int64)
                # print(label)
                tf_example = data_example(data, label)
                writer.write(tf_example.SerializeToString())
        print("Created TFRecord for test data")

    else:
        print("TFRs alerady exist...")


if __name__ == "__main__":
    create_tfrecords()
