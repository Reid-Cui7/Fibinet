import tensorflow as tf
from inputs import InputFn
from auc import AUCUtil
from ps import PS
from fibinet import config, build_fibinet

local_ps = PS(config["embedding_dim"])
train_metric = AUCUtil()
test_metric = AUCUtil()
inputs = InputFn(local_ps, config)

train_dataset = inputs.input_fn(config["train_file"], is_test=False)
test_dataset = inputs.input_fn(config["test_file"], is_test=True)

model = build_fibinet(config)
model.summary()

epochs = config["epochs"]
train_log_iter = config["train_log_iter"]
test_show_step = config["test_show_step"]


def valid_step():
    test_metric.reset()
    for batch in test_dataset:
        feature_embedding, label = batch["feature_embedding"], batch["label"]
        pred = model(feature_embedding, training=False)
        loss = tf.keras.losses.binary_crossentropy(label, pred)
        test_metric.add(loss.numpy(), label.numpy(), pred.numpy())
    print(f"Test AUC: {test_metric.calc_str()}")

def train():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        step = 0
        train_metric.reset()
        for batch in train_dataset:
            feature_embedding, label = batch["feature_embedding"], batch["label"]
            label = tf.reshape(label, (-1, 1))
            with tf.GradientTape() as tape:
                pred = model(feature_embedding, training=True)
                loss = tf.keras.losses.binary_crossentropy(label, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_metric.add(loss.numpy(), label.numpy(), pred.numpy())
            step += 1

            if step % train_log_iter == 0:
                print(f"Step {step}, Train AUC: {train_metric.calc_str()}")
                train_metric.reset()
            if step % test_show_step == 0:
                valid_step()
        print(f"End of epoch {epoch+1}, running validation...")
        valid_step()

if __name__ == "__main__":
    train()