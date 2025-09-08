import numpy as np
import tensorflow as tf

class Singleton(type):
    _instance = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instance[cls]


class PS(metaclass=Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(2025)
        self.param_server = dict()
        self.dim = embedding_dim
        print("ps init with dim:%d" % self.dim)

    def pull(self, keys):
        if hasattr(keys, 'numpy'):
            keys = keys.numpy()
        keys = keys.tolist()
        values = []
        # [batch, feature_len]
        for k in keys:
            tmp = []
            for arr in k:
                value = self.param_server.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.param_server[arr] = value
                tmp.append(value)
            values.append(tmp)
        
        return tf.convert_to_tensor(values, dtype=tf.float32)
    
    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.param_server[keys[i][j]] = values[i][j]
        return True
    
    def delete(self, keys):
        for k in keys:
            self.param_server.pop(k, None)

    def save(self, path):
        print("Total keys in ps:%d" % len(self.param_server))
        writer = open(path, "w")
        for k, v in self.param_server.items():
            writer.write(str(k) + "\t" + ",".join(['%.8f' % _ for _ in v]) + "\n")
        writer.close()

if __name__ == "__main__":
    ps_local = PS(8)
    keys = [[123, 234]]
    # 从参数服务pull keys,如果参数服务中有这个key就直接取出，若没有就随机初始取出
    res = ps_local.pull(keys)
    print(ps_local.param_server)
    print(res)
    # 经过模型迭代更新后，传入参数服务器中
    gradient = 10
    res = res - 0.01 * gradient
    ps_local.push(keys, res)
    print(ps_local.param_server)
    # 经过上述多轮的pull参数，然后梯度更新后，获得最终的key对应的向量embedding
    # 保存向量，该向量用于召回
    path = "./feature_embedding"
    ps_local.save(path)