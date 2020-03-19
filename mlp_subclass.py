import datetime
import os
import tensorflow as tf
from tensorflow import keras
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(8)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8)
    return train_dataset, test_dataset


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense2 = keras.layers.Dense(10, activation='softmax')

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


model = MyModel()
train_ds, test_ds = dataset()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# @tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(targets, predictions)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    loss = train_loss(loss)
    accuracy = train_accuracy(targets, predictions)
    return loss, accuracy


@tf.function
def test_step(inputs, targets):
    predictions = model(inputs)
    t_loss = loss_object(targets, predictions)
    test_loss(t_loss)
    test_accuracy(targets, predictions)


if __name__ == '__main__':
    s_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    train_log_dir = 'logs/train-' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    epoch = 1
    for epoch_index in range(epoch):
        batch_index = int()
        for (images, labels) in train_ds.take(3):
            tf.summary.trace_on(graph=True, profiler=True)
            loss, accuracy = train_step(images, labels)
            with train_summary_writer.as_default():
                tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=train_log_dir)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=batch_index)
                tf.summary.scalar('accuracy', accuracy, step=batch_index)
            batch_index +=1

            template = 'batch {}, Loss: {:.4f}, Accuracy: {:.4f}'
            print(template.format(batch_index, loss, accuracy))
    print('total time:', time.time() - s_time)
