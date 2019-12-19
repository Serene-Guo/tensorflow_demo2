#encoding=utf-8
import tensorflow as tf

p=tf.Variable(tf.random_normal([10,5]))#生成10*1的张量
b = tf.nn.embedding_lookup(p, [1])#查找张量中的序号为1和3的
a = tf.nn.embedding_lookup(p, [3])#查找张量中的序号为1和3的

c = a*b


d=tf.multiply(a,b)

e = tf.matmul(a,tf.transpose(b))

f=tf.reduce_sum(d)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(b))
    print(sess.run(p))
    #print(p)
    print(type(p))
    #print(type(b))
    #print(sess.run(a))
    #print(sess.run(c))
    #print(type(c))
    print(type(d))
    print(sess.run(d))
    print(sess.run(e))
    print(sess.run(f))
    print(type(e))
    print(type(f))
    print(a.shape)
    print(e.shape)
    print(f.shape)
