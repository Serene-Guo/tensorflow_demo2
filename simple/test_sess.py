import tensorflow as tf

sess=tf.Session()
din_all = tf.random_normal([2, 3, 1], mean=0.0, stddev=1.0)

b = tf.reshape(din_all, [-1, 1, 3])


print (sess.run([din_all, b]))

print (sess.run(din_all))

with tf.Session() as sess:
	print (sess.run(din_all))
	print (sess.run(b))


a = tf.constant(1)
"""
Variable：作为存储节点的变量（Variable）不是一个简单的节点，而是一副由四个子节点构成的子图：
        （1）变量的初始值——initial_value。
        （2）更新变量值的操作op——Assign。
        （3）读取变量值的操作op——read
        （4）变量操作——（a）
上述四个步骤即：首先，将initial_value赋值（Assign）给节点，存储在（a）当中，当需要读取该变量时，调用read函数取值即可
"""
b = tf.Variable(2)
"""
tf.assign(a, b) 把b的值赋值给a
"""
addop = tf.assign(b, b + 3)
c = addop + a
with tf.Session() as sess:
    """
     用来初始化 计算图中 所有global variable的 op
    """
    tf.global_variables_initializer().run()
    print (sess.run(addop))
    print (sess.run(addop))
    # 执行
    cc, bb = sess.run([addop, c])
    print(cc, bb)
    bb = sess.run(addop)
    print(bb)



state = tf.Variable(0.0,dtype=tf.float32)
one = tf.constant(1.0,dtype=tf.float32)
new_val = tf.add(state, one) # 1
update = tf.assign(state, new_val) #返回tensor， 值为new_val #update 1 # state 1
update2 = tf.assign(state, 10000) #没有fetch，便没有执行 # update2 10000 # state 10000

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run([new_val, update2, update, state]))
    print (sess.run(update.graph))

