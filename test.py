import tensorflow as tf


@tf.function
def f(x):
    if x > 0:
        import pdb
        pdb.set_trace()
        x = x + 1
    return x


tf.config.experimental_run_functions_eagerly(True)
f(tf.constant(1))
