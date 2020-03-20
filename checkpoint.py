import tensorflow as tf


# 参数保存模块：训练前删除./checkpoint目录
class checkpoint():
    def __init__(self, model, save_dir):
        self.save_dir = save_dir
        self.checkpoint = tf.train.Checkpoint(myModel=model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.save_dir, max_to_keep=3)

    def save_model(self, step):
        if step % 100 == 0:
            path = self.manager.save(checkpoint_number=step)  # 用manager来保存checkpoint文件
            print("model saved to %s" % path)

    def restore_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.save_dir))  # 从文件恢复最新的模型参数
