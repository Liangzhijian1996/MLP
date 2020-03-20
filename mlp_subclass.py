from tensorflow import keras
import tensorflow as tf


class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model, self).__init__()
        self.block_1 = keras.layers.Dense(64, activation='relu', name='dense_1')
        self.block_2 = keras.layers.Dense(64, activation='relu', name='dense_2')
        self.block_3 = keras.layers.Dense(10, name='predictions')

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        outputs = self.block_3(x)
        return outputs


optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 数据处理部分
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
y_train = y_train.astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(64)


def train():
    # 借用 compile+fit 指定模型各层形状
    model = my_model()
    # model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    #               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # model.fit(x_train, y_train, batch_size=64, epochs=1)

    # 进行自定义训练
    for epoch in range(1):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 200 == 0:
                print('Training loss at step %s: %s' % (step, float(loss_value)))
    # model.save('saved/my_model_2', save_format='tf')
    model.save_weights('saved/my_weights', save_format='tf')


def test():
    new_model = my_model()
    new_model.load_weights('saved/my_weights')

    # new_model = keras.models.load_model('saved/my_model_2')

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset.take(1)):
        y_prep = new_model(x_batch_train, training=True)
        loss = loss_fn(y_batch_train, y_prep)
        print(loss.numpy())


# train()
# test()
