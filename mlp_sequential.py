import tensorflow as tf
from data_loader import MNISTLoader
import numpy as np


def mlp_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])
    model.summary()
    tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)
    return model


def train1():
    data_loader = MNISTLoader()
    model = mlp_model()

    print("——————训练环节——————")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs=4, batch_size=50)

    print("——————测试环节——————")
    print(model.evaluate(data_loader.test_data, data_loader.test_label))


def train2():
    data_loader = MNISTLoader()
    model = mlp_model()

    print("——————训练环节——————")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 实例化一个优化器
    for batch_index in range(1, 6000 + 1):
        x, y = data_loader.get_batch(50)  # 加载一个batch数据
        with tf.GradientTape() as tape:  # GradientTape：用于为后面计算梯度服务
            y_pred = model(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)  # 计算梯度，自动释放GradientTape资源
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 优化

    print("——————测试环节——————")
    y_pred = np.argmax(model.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))

