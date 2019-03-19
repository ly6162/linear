import tensorflow as tf
import ML_liner.inference as infer
import tensorflow_liner.train as train

def inference_custom(input):
    weight = tf.Variable([0], dtype=tf.float32)
    bias = tf.Variable([0], dtype=tf.float32)
    ...
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, train.model_tf_custom)
        w=sess.run(weight)
        b=sess.run(bias)

        print('W: %s b: %s ' % (w, b))
        #推論関数を呼び出す
        infer.eval_graph("Tensorflow",input,w,b)

def inference_TF(input):
    weight = tf.Variable([0], dtype=tf.float32)
    bias = tf.Variable([0], dtype=tf.float32)

    x = tf.placeholder(tf.float32)
    ...
    y = weight * x + bias

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, train.model_tf_custom)

        out=sess.run(y,{x:input})
        print(out)
        #推論関数を呼び出す
        #print('W: %s b: %s ' % (w, b))
        #w=sess.run(weight)
        #b=sess.run(bias)

        #print(w * input + b)
def inference_TF2(input):
    weight = tf.get_variable("W", [1], dtype=tf.float64)
    bias = tf.get_variable("b", [1], dtype=tf.float64)

    x = tf.placeholder(tf.float32)
    ...
    y = weight * x + bias

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, model_path)

        out=sess.run(y,{x:input})
        print(out)
        #推論関数を呼び出す
        #print('W: %s b: %s ' % (w, b))
        #w=sess.run(weight)
        #b=sess.run(bias)

        #print(w * input + b)
if __name__ == "__main__":
    inference_custom(0.1)
    inference_TF(0.1)
