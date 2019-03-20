import tensorflow as tf

v = tf.Variable(4.0)
x = tf.placeholder(tf.float32, shape=(None))
y = tf.add(v, x)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # SavedModel保存
    tf.saved_model.simple_save(
        sess, "/tmp/saved_model/model",
        inputs={"x": x}, outputs={"y": y})

with tf.Session() as sess:
    # SavedModel読み込み
    tf.saved_model.load(
        sess, [tf.saved_model.tag_constants.SERVING], "/tmp/saved_model/model")