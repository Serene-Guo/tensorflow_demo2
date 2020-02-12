import tensorflow as tf

a= 0
b = 400 
c = tf.add(a,b, name="Add")


d = tf.convert_to_tensor([3, 1, 203])

off = tf.convert_to_tensor([2,2,2])
print (d)


mod_size = 200

offset = 2
tensor = d

new_tensor_0 = tf.mod(c - offset, mod_size) + offset if mod_size  else c


#is_notNone = tf.cast((mod_size is not None), tf.bool)

#if_1 = (mod_size is not None) and tf.greater_equal(c, offset)
#print (if_1)


#if_2 = (mod_size is not None) and tf.greater_equal(tensor, mod_size + offset)
#if_3 = tf.cast(if_2, tf.bool)

#new_tensor = tf.cond(tf.cast(if_1, tf.bool),  lambda: tf.mod(c - offset, mod_size) + offset,  lambda: c)
#new_tensor_2 = tf.cond( if_3,lambda:tf.mod(tensor, mod_size), lambda: tensor)

### new_tensor = tf.cond((mod_size is not None) and tf.greater_equal(tensor, offset),  lambda: tf.mod(tensor - offset, mod_size) + offset,  lambda: tensor) ## is mod_size is None, error.  
#  tensor is [2048,1] 

#new_tensor = tf.cond(tf.greater_equal(tensor, offset), lambda: tf.mod(tensor - offset, mod_size) + offset, lambda: tensor)

e = c < offset
f = tensor < offset

print (e)
print (f)


new_tensor = tf.cond(c<offset,lambda:tensor, lambda: tf.mod(tensor - offset, mod_size) + offset)
new_tensor_2 = tf.where(tensor<offset, x=tensor, y=tf.mod(tensor-offset, mod_size) + offset)


#Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()

writer = tf.summary.FileWriter("../graphs/", sess.graph)

#print (sess.run(new_tensor_0))
print (sess.run(d))
print (sess.run(new_tensor))
print (sess.run(new_tensor_2))

writer.close()
#sess.close()
