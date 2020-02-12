import tensorflow as tf


c = tf.convert_to_tensor(400)
d = tf.convert_to_tensor([3, 1, 203])


mod_size = 200

offset = 2
tensor = d
#tensor = c

if_1 = (mod_size is not None) and tf.greater_equal(tensor, offset)
print (if_1)

if_2 = tf.cast(if_1, tf.bool)
print (if_2)



#new_tensor_cond = tf.cond(tf.cast(if_1, tf.bool),  lambda: tf.mod(tensor - offset, mod_size) + offset,  lambda: tensor) ## is mod_size is None, error.  
#  tensor is [2048,1] 

e = c < offset
f = tensor < offset

print (e)
print (f)


new_tensor_where = tf.where(tensor<offset, x=tensor, y=tf.mod(tensor-offset, mod_size) + offset) if mod_size else tensor


#Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()

writer = tf.summary.FileWriter("../graphs/", sess.graph)

#print (sess.run(new_tensor_cond))

print (sess.run(tensor))
print (sess.run(new_tensor_where))

writer.close()
#sess.close()
