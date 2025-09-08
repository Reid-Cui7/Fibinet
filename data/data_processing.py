import random
import mmh3
import tensorflow as tf


names = [
    'id', 'date', 'user_id', 'product', 'campaign_id', 'webpage_id', 'product_category_id',
    'user_group_id', 'gender', 'age_level', 'user_depth', 'var_1'
]

def to_hash(s, is_test=True):
    seed = 17
    tmp = mmh3.hash(s, seed=seed, signed=False)
    return tmp

def ext_train_test_to_hash():
    raw_data_train = "./raw/train.csv"
    raw_data_val = "./raw/val.csv"

    raw_data_lines = open(raw_data_train, encoding="utf-8")
    raw_data_write_to_hash = open(raw_data_train + ".hash", 'w')
    val_data_write_to_hash = open(raw_data_val + ".hash", 'w')
    for i, line in enumerate(raw_data_lines):
        if i == 0:
            continue
        tmp = to_hash(line, False)
        if tmp is not None:
            raw_data_write_to_hash.write(tmp, "\n")
        raw_data_write_to_hash.close()
        val_data_write_to_hash.close()
        raw_data_lines = open(raw_data_val, encoding="utf-8")
        raw_data_write_to_hash = open(raw_data_val + ".hash", 'w')
        for i, line in enumerate(raw_data_lines):
            if i == 0:
                continue
            tmp = to_hash(line, True)
            if tmp is not None:
                val_data_write_to_hash.write(tmp + "\n")
        raw_data_write_to_hash.close()

def get_tfrecords_example(feature, label):
    tfrecord_features = {
        "feature": tf.train.Feature(int64_list=tf.train.Int64List(value=feature)),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(features=tf.train.Features(feature=tfrecord_features))


def to_tf_records():
    raw_data_train= "./raw/train.csv.hash"
    raw_data_val = "./raw/val.csv.hash"
    writer = tf.io.TFRecordWriter("./train/train.csv.hash.tfrecords")
    lines = open(raw_data_train)
    for i, line in enumerate(lines):
        line_arr = line.strip().split(",")
        feature = []
        label = []
        for j, feat in enumerate(line_arr):
            if j == len(line_arr) - 1:
                label.append(float(feat))
            else:
                feature.append(int(feat))
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
    print("Process To tfrecord File: %s End" % raw_data_train)
    writer.close()

    writer = tf.io.TFRecordWriter("./val/val.csv.hash.tfrecords")
    lines = open(raw_data_val)
    for i, line in enumerate(lines):
        line_arr = line.strip().split(",")
        feature = []
        label = []
        for j, feat in enumerate(line_arr):
            if j == len(line_arr) - 1:
                label.append(float(feat))
            else:
                feature.append(int(feat))
        example = get_tfrecords_example(feature, label)
        writer.write(example.SerializeToString())
    print("Process To tfrecord File: %s End" % raw_data_val)
    writer.close()


def main():
    # step01: 处理训练数据, hash
    # ext_train_test_to_hash()
    # step02: tfrecord
    to_tf_records()


if __name__ == "__main__":
    main()