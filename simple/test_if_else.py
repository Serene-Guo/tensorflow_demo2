import tensorflow as tf

a= 0
b = 400 
c = tf.add(a,b, name="Add")


d = tf.convert_to_tensor([3, 1, 203])


mod_size = 200
mod_size = None
offset = 2


tensor = d
new_tensor_0 = tf.mod(tensor - offset, mod_size) + offset if mod_size  else tensor


sess = tf.Session()
writer = tf.summary.FileWriter("../graphs/", sess.graph)

print (sess.run(tensor))
print (sess.run(new_tensor_0))

writer.close()
#sess.close()
