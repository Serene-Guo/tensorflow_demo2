import tensorflow as tf

p=tf.Variable(tf.random_normal([10,1]))#生成10*1的张量
b = tf.nn.embedding_lookup(p, [1, 3])#查找张量中的序号为1和3的
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    #print(c)
    print(sess.run(p))
    print(p)
    print(type(p))
