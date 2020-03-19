import tensorflow as tf
import numpy as np


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        # 重要：从本地加载数据
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]  # 60000,10000

    def get_batch(self, batch_size):
        # 随机生成batch_size个介于0-60000之间的随机数
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)  # 0,60000,batch_size
        # 返回batch_size个随机数在60000中对应位置的训练数据和标签
        return self.train_data[index, :], self.train_label[index]


