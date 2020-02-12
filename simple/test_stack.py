import tensorflow as tf


c = tf.convert_to_tensor(400)
d = tf.convert_to_tensor([[1,1,1], [3,3,3], [5,5,5], [7,7,7]])
e = tf.convert_to_tensor([[2,2,2], [4,4,4], [6,6,6], [8,8,8]])

tensors = [d, e]


a = tf.stack(tensors, axis=1, name="tmp")


bi_reduce_sum = tf.reduce_sum(a, axis=1)
bi_sum_square = tf.square(bi_reduce_sum, name='bi_sum_square')

bi_square = tf.square(a)
bi_square_sum = tf.reduce_sum(bi_square, axis=1, name='bi_square_sum')



#Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()

writer = tf.summary.FileWriter("../graphs/", sess.graph)

#print (sess.run(new_tensor_cond))

print (d)
print ("after stack:")
print (a)
print (sess.run(a))

print ("a _reduce_sum, axis=1")
print (bi_reduce_sum)
print (sess.run(bi_reduce_sum))

print ("bi_sum_square, tf.square")
print (bi_sum_square)
print (sess.run(bi_sum_square))


print ("bi_square")
print (bi_square)
print (sess.run(bi_square))

print ("bi_square_sum")
print (bi_square_sum)
print (sess.run(bi_square_sum))

print (sess.run(bi_sum_square - bi_square_sum))
writer.close()
#sess.close()
