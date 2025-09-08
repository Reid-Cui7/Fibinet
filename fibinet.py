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
    "reduction_ratio": 0.5,
    "bilinear_type": "all",
    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "./data/train",
    "test_file": "./data/val",
    "saved_embedding": "./data/saved_dnn_embedding",
    "max_steps": 10000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "fibinet",

    "saved_pb": "./data/saved_model",

    "input_tensor": ["input_tensor"],
    "output_tensor": ["output_tensor"]
}


def nn_tower(
        name, nn_input, hidden_units,
        activation=tf.nn.relu, use_bias=False,
        l2=0.0):
    out = nn_input
    for i, num in enumerate(hidden_units):
        out = tf.layers.dense(
            out,
            units=num,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2),
            use_bias=use_bias,
            activation=activation,
            name=name + "/layer_" + str(i),
        )
    return out

# SENet
# 想通过控制scale的大小，
# 把重要的特征增强，不重要的特征减弱，从而让提取的特征指向性更强
def squeeze_excitation_layer(name_scope, inputs, ratio=16):
    """
    in: [batch, field, embed]
    out: [batch, field, embed]
    """
    with tf.compat.v1.variable_scope(name_scope):
        inputs_shape = inputs.get_shape().as_list()
        squeeze = tf.reduce_sum(inputs, axis=2) # [batch, field]
        # 按比例压缩
        excitation = nn_tower(name="ex01", nn_input=squeeze,
                            hidden_units=[inputs_shape[1] // ratio], use_bias=True)
        excitation = nn_tower(name="ex02", nn_input=excitation,
                            hidden_units=[inputs_shape[1]], use_bias=True, activation=tf.nn.sigmoid)
        excitation = tf.tile(tf.reshape(
            excitation, [-1, inputs_shape[1], 1]), [1, 1, inputs_shape[2]])
        scale = inputs * excitation
    return scale

def bi_linear_interaction_layer(name_score, inputs, bilinear_type):
    """
    in: [batch, field, embed]
    out: [batch, field * embed]
    """
    with tf.compat.v1.variable_scope(name_score):
        inputs_shape = inputs.get_shape().as_list()
        embedding_size = inputs_shape[2]
        inputs_list = tf.split(inputs, inputs_shape[1], axis=1)
        list_num = len(inputs_list)
        if bilinear_type == "all":
            w = tf.get_varibale(
                "bilinear_weight",
                [embedding_size, embedding_size],
                initializer=tf.glorot_normal_initializer(seed=2025)
            )
            p = [tf.multiply(tf.tensordot(v_i, w, axes=(-1, 0)), v_j) 
                            for v_i, v_j in itertools.combinations(inputs_list, 2)]
        elif bilinear_type == "each":
            w_list = []
            for i in range(list_num - 1):
                w_list.append(tf.get_variable(
                    "bilinear_weight_" + str(i),
                    [embedding_size, embedding_size],
                    initializer=tf.glorot_normal_initializer(seed=2025)
                ))
                p = [tf.multiply(tf.tensordot(inputs_list[i], w_list[i], axes=(-1, 0)), inputs_list[j])
                    for i, j in itertools.combinations(range(list_num), 2)]
        elif bilinear_type == "interaction":
            w_list = []
            for i, j in itertools.combinations(range(list_num), 2):
                w_list.append(tf.get_variable(
                    "bilinear_weight_" + str(i) + "_" + str(j),
                    [embedding_size, embedding_size],
                    initializer=tf.glorot_normal_initializer(seed=2025)
                ))
                p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                    for v, w in zip(itertools.combinations(inputs_list, 2), w_list)]
        else:
            raise ValueError("bilinear_type must in ['all', 'each', 'interaction']")

        p = tf.concat(p, axis=1)
        return p

def fibinet_model(inputs, is_test=False):
    # 获取feature和label feature: [batch, f_nums, weight_dim]
    input_embedding = tf.reshape(inputs["feature_embedding"],
                                shape=[-1, config["feature_len"], config["embedding_dim"]])
    senet_embedding = squeeze_excitation_layer(name_scope="senet", inputs=input_embedding, ratio=config["reduction_ratio"])

    senet_bilinear_out_ = bi_linear_interaction_layer(
        name_score="senet_bilinear",
        inputs=senet_embedding,
        bilinear_type=config["bilinear_type"]
    )
    senet_bilinear_out_ = bi_linear_interaction_layer(
        name_score="input_bilinear",
        inputs=input_embedding,
        bilinear_type=config["bilinear_type"]
    )
    out_ = tf.concat([senet_bilinear_out_, senet_bilinear_out_], axis=1)

    out = nn_tower(
        "dnn_hidden",
        out_, config["dnn_hidden_units"],
        use_bias=True,
        activation=config["activation_function"],
        l2 = config["l2_reg"]
    )
    out_ = nn_tower("out", out_, [1], activation=None)
    out_ = tf.reshape(out_, [-1])
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=out_, labels=inputs["label"][:, 0])
    )
    if is_test:
        tf.add_to_collections("input_tensor", input_embedding)
        tf.add_to_collections("output_tensor", out_)
    net_dic = {
        "loss": loss,
        "ground_truth": inputs["label"][:, 0],
        "prediction": out_
    }
    return net_dic

# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.compat.v1.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = fibinet_model(inputs, is_test)

        loss = net_out_dic["loss"]

        result["out"] = net_out_dic
        if is_test:
            return result

        # ps - sgd
        emb_grad = tf.gradients(
            loss, [inputs["feature_embedding"]], name="feature_embedding")[0]

        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - config['learning_rate'] * emb_grad

        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        # net - sgd
        tvars1 = tf.trainable_variables()
        grads1 = tf.gradients(loss, tvars1)
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=config['learning_rate'],
            use_locking=True)
        train_op = opt.apply_gradients(zip(grads1, tvars1))
        result["train_op"] = train_op

        return result
    
########### tf2改的
class SENetLayer(layers.Layer):
    def __init__(self, field_num, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.field_num = field_num
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        reduced_dim = max(1, int(self.field_num * self.reduction_ratio))
        self.fc1 = layers.Dense(reduced_dim, activation='relu')
        self.fc2 = layers.Dense(self.field_num, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [batch, field, embed]
        # 要先去掉最后一个维度不能keep 不然处理逻辑不对
        squeeze = tf.reduce_sum(inputs, axis=2)  # [batch, field]
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
        self.field_num = input_shape[1]
        self.embed_dim = input_shape[2]
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
    # SENet
    senet_embedding = SENetLayer(feature_len, reduction_ratio)(feature_embedding)
    senet_bilinear_out = BiLinearInteractionLayer(bilinear_type)(senet_embedding)
    input_bilinear_out = BiLinearInteractionLayer(bilinear_type)(feature_embedding)
    out = layers.Concatenate(axis=1)(senet_bilinear_out, input_bilinear_out)


if __name__ == "__main__":
    build_fibinet(config)