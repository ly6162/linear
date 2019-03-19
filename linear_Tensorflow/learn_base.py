# 引入 tensorflow 模块
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 定数Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)

# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])

# 打印上面创建的几个 Tensor
print(t0)
print(t1)
print(t2)
print(t3)
sess = tf.Session()
print(sess.run(t0))
out=sess.run(t1)
print(out)
print(sess.run(t2))

# 创建两个常量节点
node1 = tf.constant(3.02)
node2 = tf.constant(4.8)
str1="my "
str2="is liu"
str3=str1+str2
print(str3)
tf_str1=tf.constant(str1,dtype=tf.string)
tf_str2=tf.constant(str2,dtype=tf.string)
tf_str3=tf_str1+tf_str2
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder = node1 + node2
# 打印一下 adder 节点
print(adder)
# 打印 adder 运行后的结果
with tf.device('/device:GPU:0'):
    sess1 = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))
    #for i in range(10000):
        #print(sess1.run(adder))
    print(sess1.run(tf_str3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 创建一个 adder 节点，对上面两个节点执行 + 操作
adder_node = a + b
# 打印三个节点
print(a)
print(b)
print(adder)
# 运行一下，后面的 dict 参数是为占位 Tensor 提供输入数据
sess = tf.Session()
print(sess.run(adder_node, {a: 30, b: 4.5}))
print(sess.run(adder_node, {a: [10, 3], b: [2, 4]}))