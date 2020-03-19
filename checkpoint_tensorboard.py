import datetime
import tensorflow as tf
import numpy as np


# 参数保存模块：训练前删除./checkpoint目录
class checkpoint():
    def __init__(self, model):
        self.save_dir = './checkpoint'
        self.checkpoint = tf.train.Checkpoint(myModel=model)
        # 用manager管理checkpoint文件,restore_model时其实没用到这句
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.save_dir, max_to_keep=5)

    def save_model(self, batch_index):
        if batch_index % 100 == 0:  # 每隔100个Batch保存一次模型参数到文件
            path = self.manager.save(checkpoint_number=batch_index)  # 用manager来保存checkpoint文件
            print("model saved to %s" % path)

    def restore_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.save_dir))  # 从文件恢复最新的模型参数


# 可视化模块：训练前删除log_dir目录
# 控制台命令:tensorboard --logdir=/home/lzj/PycharmProject/MLP/tensorboard/events.out.tfevents.1584277278.lzj.18800.64.v2
class tensorboard():
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.train_log_dir = './logs/train' + self.current_time
        self.test_log_dir = './logs/test' + self.current_time
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        tf.summary.trace_on(profiler=True)  # 开启Trace：用于记录计算图Graph、操作信息设备使用等Profile

    def save_scalar(self, scalar, step):
        with self.train_summary_writer.as_default():  # 将当前scalar项目：loss值写入记录器
            tf.summary.scalar("loss", scalar, step=step)

    def save_image(self, img, step):
        with self.train_summary_writer.as_default():  # 将当前scalar项目：loss值写入记录器
            img = np.reshape(img[0], (-1, 64, 64, 3))
            tf.summary.image('input', img, step=step)

    def save_grasp_profile(self):
        with self.train_summary_writer.as_default():  # 保存Trace信息到文件（可选）
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.train_log_dir)