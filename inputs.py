import os
import tensorflow as tf


class InputFn:
    def __init__(self, local_ps, config):
        self.feature_len = config["feature_len"]
        self.label_len = config["label_len"]
        self.n_parse_threads = config["n_parse_threads"]
        self.shuffle_buffer_size = config["shuffle_buffer_size"]
        self.prefetch_buffer_size = config["prefetch_buffer_size"]
        self.batch = config["batch"]
        self.local_ps = local_ps


    def input_fn(self, data_dir, is_test=False):

        def _parse_example(example):
            features = {
                "feature": tf.io.FixedLenFeature(self.feature_len, tf.int64),
                "label": tf.io.FixedLenFeature(self.label_len, tf.float32)
            }
            return tf.io.parse_single_example(example, features)
        
        def _get_embedding(parsed):
            keys = parsed["feature"]
            keys_array = tf.py_function(self.local_ps.pull, [keys], tf.float32)
            result = {
                "feature": parsed["feature"],
                "label": parsed["label"],
                "feature_embedding": keys_array
            }
            return result

        file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        dataset = tf.data.Dataset.list_files(file_list)
        dataset = dataset.interleave(lambda _: tf.data.TFRecordDataset(_), cycle_length=1)
        dataset = dataset.map(_parse_example, num_parallel_calls=self.n_parse_threads)        
        if not is_test:
            dataset.shuffle(self.shuffle_buffer_size)
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch, drop_remainder=True)
        dataset = dataset.map(_get_embedding, num_parallel_calls=self.n_parse_threads)
        dataset = dataset.prefetch(self.prefetch_buffer_size)

        return dataset

if __name__ == "__main__":
    from ps import PS
    from fibinet import config
    local_ps = PS(8)
    inputs = InputFn(local_ps, config)
    data_dir = "./data/train"
    train_inputs = inputs.input_fn(data_dir, is_test=False)
    for batch in train_inputs:
        print(batch)
        break