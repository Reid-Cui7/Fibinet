import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tfrecord_path", type=str, default="./train/train.csv.hash.tfrecords", help="tfrecord file path")
args = parser.parse_args()

path = args.tfrecord_path

def read_tf():
    for serialized_example in tf.compat.v1.io.tf_record_iterator(path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        feature = example.features.feature["feature"].int64_list.value
        label = example.features.feature["label"].float_list.value
        print("feature:", feature)
        print("label:", label)
        break

read_tf()