
from inputs import InputFn
from auc import AUCUtil
from ps import PS
from fibinet import *

local_ps = PS(config["embedding_dim"])
train_metric = AUCUtil()
test_metric = AUCUtil()
inputs = InputFn(local_ps, config)

# model
train_inputs = inputs.input_fn(config["train_file"], is_test=False)
train_dic = setup_graph(train_inputs, is_test=False)