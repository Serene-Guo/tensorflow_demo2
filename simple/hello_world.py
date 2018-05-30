import tensorflow as tf

a= 2
b = 3
c = tf.add(a,b, name="Add")

print (c)

#Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()

writer = tf.summary.FileWriter("../graphs/", sess.graph)
print (sess.run(c))

writer.close()
#sess.close()
