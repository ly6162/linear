import tensorflow as tf
import time,os
import numpy as np
import data
from hparam import hparam
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

model_tf_custom="../data/model_tf_custom/tf_line_model.ckpt"
model_tf_highAPI="../data/model_tf_highAPI/2tf_line_model.ckpt"

x_input, y_input = data.load()
class linear():
    def __init__(self):
        # W,bの変数を定義
        self.W = tf.Variable([.0], dtype=tf.float32)
        self.b = tf.Variable([.0], dtype=tf.float32)

        # x,yの入力データの変数を定義
        self.x = tf.placeholder(tf.float32) #学習データ
        self.y = tf.placeholder(tf.float32) #教師データ

        # 線形モデル定義
        linear_model = self.W * self.x + self.b

        # 損失関定義
        self.loss = tf.reduce_sum(tf.square(linear_model - self.y))
        # 学習率
        optimizer = tf.train.GradientDescentOptimizer(hparam.learning_rate)
        #最適化方式（小さい方式）
        self.train = optimizer.minimize(self.loss)

#tensorflowの# 従来の APIを利用して開発した、特徴は、計算プロセスは明確的に見えます。
def train_custom():

    model=linear()
    # Session を定義、
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))

    # 計算グラフィックを初期化
    init = tf.global_variables_initializer()
    sess.run(init)

    start = time.time()
    for i in range(hparam.steps):

        sess.run(model.train, {model.x: x_input, model.y: y_input})
        if i%hparam.log_step==0:
            print("step:%s"%i, sess.run(model.loss,{model.x: x_input, model.y: y_input}))
    print('train time: %.5f' % (time.time()-start))

    # Sessionごとを保存
    saver = tf.train.Saver()
    saver.save(sess, model_tf_custom)
    print(model_tf_custom)
    # 学習結果を確認
    print('weight: %s bias: %s loss: %s' % (sess.run(model.W), sess.run(model.b), sess.run(model.loss,{model.x: x_input, model.y: y_input})))

#tensorflowの高級 APIを利用して開発した、特徴は、学習と推論はより簡単になった、それらの処理はtensorflow内部側やってくれます。
#この例は開発中、推論の部分は未完成です。
def train_high_API():
    def model_fn(features, labels, mode):
        # 計算式
        W = tf.get_variable("W", [1], dtype=tf.float64)
        b = tf.get_variable("b", [1], dtype=tf.float64)
        y = W * features['x'] + b

        loss = tf.reduce_sum(tf.square(labels-y))
        #tf.reduce_sum(tf.square(linear_model - self.y))
        # 学習子グラフィック
        global_step = tf.train.get_global_step()
        optimizer = tf.train.GradientDescentOptimizer(hparam.learning_rate)
        train = tf.group(optimizer.minimize(loss),
                         tf.assign_add(global_step, 1))
        # EstimatorSpecをと通じて学習
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y,
            loss=loss,
            train_op=train
            )

    def train():
        # 学習などの設定
        estimator = tf.estimator.Estimator(model_fn=model_fn,model_dir="/home/liu/tf/test")

        x_eavl = np.array([2., 5., 7., 9.])
        y_eavl = np.array([7.6, 17.2, 23.6, 28.8])
        for i in range(3):
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_input}, y_input, batch_size=hparam.batch_size, num_epochs=hparam.steps, shuffle=True)
            estimator.train(input_fn=train_input_fn)

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_eavl}, y_eavl, batch_size=hparam.batch_size, num_epochs=hparam.log_step, shuffle=False)

            train_metrics = estimator.evaluate(input_fn=train_input_fn)
            print("train metrics: %r" % train_metrics)

            #eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
            #print("eval metrics: %s" % eval_metrics)
    train()

if __name__ == "__main__":
    train_high_API()
    #train_custom()