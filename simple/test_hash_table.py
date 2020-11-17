import tensorflow as tf
 
print(tf.__version__)
list_arr = [9, 8, 6, 5]
value_arr = [0, 1, 2, 3]
tf_look_up = tf.constant(list_arr, dtype=tf.int64)
tf_value_arr = tf.constant(value_arr, dtype=tf.int64)
 
table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf_look_up, tf_value_arr), 0)
ph_vals = tf.constant([8, 5], dtype=tf.int64)
ph_idx = table.lookup(ph_vals)
 
with tf.Session() as sess:
	sess.run(tf.tables_initializer())
	sess.run(tf.initialize_all_variables())
	res = sess.run(ph_idx)
	print(res)

