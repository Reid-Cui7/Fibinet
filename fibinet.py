import itertools
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers




config = {
    "feature_len": 10,
    "embedding_dim": 5,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,
    "learning_rate": 0.01,
    "reduction_ratio": 3,
    "bilinear_type": "all",
    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "./data/train",
    "test_file": "./data/val",
    "saved_embedding": "./data/saved_dnn_embedding",
    "epochs": 2,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "fibinet",

    "saved_pb": "./data/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


class SENetLayer(layers.Layer):
    def __init__(self, field_num, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.field_num = field_num
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        reduced_dim = max(1, int(self.field_num // self.reduction_ratio))
        self.fc1 = layers.Dense(reduced_dim, activation='relu')
        self.fc2 = layers.Dense(self.field_num, activation='relu')
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [batch, field, embed]
        # 要先去掉最后一个维度不能keep 不然处理逻辑不对
        squeeze = tf.reduce_mean(inputs, axis=2)  # [batch, field]
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)  # [batch, field]
        excitation = tf.expand_dims(excitation, axis=-1)  # [batch, field, 1]
        scale = inputs * excitation  # [batch, field, embed]
        return scale


class BiLinearInteractionLayer(layers.Layer):
    def __init__(self, bilinear_type="all", **kwargs):
        super().__init__(**kwargs)
        self.bilinear_type = bilinear_type

    def build(self, input_shape):
        _, self.field_num, self.embed_dim = input_shape
        if self.bilinear_type == "all":
            self.w = self.add_weight(
                shape=(self.embed_dim, self.embed_dim),
                initializer='glorot_normal',
                trainable=True,
                name='bilinear_weight'
            )
        elif self.bilinear_type == "each":
            self.w_list = [
                self.add_weight(
                    shape=(self.embed_dim, self.embed_dim),
                    initializer='glorot_normal',
                    trainable=True,
                    name='bilinear_weight_' + str(i)
                ) for i in range(self.field_num)    
            ]
        elif self.bilinear_type == "interaction":
            self.w_list = [
                self.add_weight(
                    shape=(self.embed_dim, self.embed_dim),
                    initializer='glorot_normal',
                    trainable=True,
                    name='bilinear_weight_' + str(i) + "_" + str(j)
                ) for i, j in itertools.combinations(range(self.field_num), 2)
            ]

    def call(self, inputs):
        inputs_list = tf.split(inputs, self.field_num, axis=1)
        inputs_list = [tf.squeeze(x, axis=1) for x in inputs_list] # [batch, embed]
        if self.bilinear_type == "all":
            p = [tf.multiply(tf.matmul(v_i, self.w), v_j)
                    for v_i, v_j in itertools.combinations(inputs_list, 2)
                ]
        elif self.bilinear_type == "each":
            p = [tf.multiply(tf.matmul(inputs_list[i], self.w_list[i]), inputs_list[j])
                    for i, j in itertools.combinations(range(self.field_num), 2)
                ]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.matmul(v[0], w), v[1])
                    for v, w, in zip(itertools.combinations(inputs_list, 2), self.w_list)
                ]
        else:
            raise ValueError("bilinear_type not in (all, each, interaction)")
        return tf.concat(p, axis=1)
    

def build_fibinet(config):
    feature_len = config["feature_len"]
    embedding_dim = config["embedding_dim"]
    dnn_hidden_units = config["dnn_hidden_units"]
    activation_function = config["activation_function"]
    dnn_l2 = config["dnn_l2"]
    reduction_ratio = config["reduction_ratio"]
    bilinear_type = config["bilinear_type"]
    # Inputs
    feature_embedding = layers.Input(shape=(feature_len, embedding_dim), name="feature_embedding")
    label = layers.Input(shape=config["label_len"], name="label")
    # SENet + Bilinear
    senet_embedding = SENetLayer(feature_len, reduction_ratio)(feature_embedding)
    senet_bilinear_out = BiLinearInteractionLayer(bilinear_type)(senet_embedding)
    input_bilinear_out = BiLinearInteractionLayer(bilinear_type)(feature_embedding)
    out = layers.Concatenate(axis=1)([senet_bilinear_out, input_bilinear_out])
    # DNN
    for i, unit in enumerate(dnn_hidden_units):
        out = layers.Dense(unit, 
                            activation=activation_function,
                            kernel_regularizer=regularizers.l2(dnn_l2),
                            name=f"dnn_hidden_{i}")(out)
    logits = layers.Dense(1, activation=None, name="logits")(out)
    output = layers.Activation('sigmoid', name="output_tensor")(logits)
    # model
    model = Model(inputs=[feature_embedding], outputs=output)
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    return model


if __name__ == "__main__":
    build_fibinet(config)