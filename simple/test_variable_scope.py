import tensorflow as tf



def get_tensor():
    with tf.variable_scope(name_or_scope="scope_1", reuse=tf.AUTO_REUSE):
        a = tf.convert_to_tensor([[1],[2],[3]])
        return a

def process_tensor(tensor_input):
    with tf.variable_scope(name_or_scope="scope_2", reuse=tf.AUTO_REUSE):
        b = tensor_input * 2
        return b

def get_tensor_2(input_tensor):
    with tf.variable_scope(name_or_scope="scope_3", reuse=tf.AUTO_REUSE):
         initializer = tf.random_normal_initializer()
         w = tf.get_variable("docid_table", shape=(10, 8),
                initializer=initializer,
                 dtype=tf.float32,
                 )
         w = tf.Print(w, [w], message='w_random',summarize=20)
         #w = w +1
     
         feature = tf.nn.embedding_lookup(w, input_tensor)
         return feature

def preprocess_embedding(tensor, feature_size, embedding_size, scope_name, weight_name, mod_size=None, offset=0):
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE):
        initializer = tf.random_normal_initializer()
        new_feature_size = mod_size + offset if mod_size else feature_size
        w = tf.get_variable(weight_name, shape=(new_feature_size, embedding_size),
                                        initializer=initializer,
                                        dtype=tf.float32,
                                    )    
        new_tensor = tf.where(tensor<offset, x=tensor, y=tf.mod(tensor-offset, mod_size) + offset) if mod_size else tensor

        #feature = tf.nn.embedding_lookup(w, tf.mod(tensor, mod_size) if mod_size else tensor)
        feature = tf.nn.embedding_lookup(w, new_tensor)
        return feature

def preprocess_weighted_avg_pooling_fixedsize(tensor, tensor_weight, tensor_length, scope_name):
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE):
        tensor_weight = tf.add(tensor_weight, 0.000001)
        weight_sum = tf.reduce_sum(tensor_weight, axis=-1, keepdims=True)
        weight_sum = tf.Print(weight_sum, [weight_sum, weight_sum.shape], message='weight_sum',summarize=20)
        tensor_weight = tf.div(tensor_weight, weight_sum)
        tensor_weight = tf.Print(tensor_weight, [tensor_weight, tensor_weight.shape], message='after_div',summarize=20)
        if len(tensor.get_shape().as_list()) > len(tensor_weight.get_shape().as_list()):
            tensor_weight = tf.expand_dims(tensor_weight, -1)
        tensor_weight = tf.Print(tensor_weight, [tensor_weight, tensor_weight.shape], message='expan_dims',summarize=20)
        feature = tf.multiply(tensor, tensor_weight)
        feature = tf.Print(feature, [feature, feature.shape], message='multiply(tensor, tensor_weight)',summarize=20)
        feature = tf.reduce_sum(feature, axis=1)
        feature = tf.Print(feature, [feature, feature.shape], message='after_reduce_sum',summarize=20)
        feature = feature / tf.cast(tensor_length, tf.float32)
        return feature

def preprocess_sum_pooling_fixedsize(tensor, tensor_weight, tensor_length, scope_name):
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE):       
        mask = tf.sequence_mask(tensor_length, tf.shape(tensor)[1], dtype=tf.float32)
        mask = tf.Print(mask, [mask, mask.shape], message='sequence_mask',summarize=20)
        mask = tf.reshape(mask, [-1, tf.shape(tensor)[1], 1])
        mask = tf.Print(mask, [mask, mask.shape], message='reshape mask',summarize=20)
        feature = tf.multiply(tensor, mask)
        feature = tf.Print(feature, [feature, feature.shape], message='multiply(tensor, mask)',summarize=20)
        feature = tf.reduce_sum(feature, axis=1)
        return feature


#
a = get_tensor()
b = process_tensor(a)
#def preprocess_embedding(tensor, feature_size, embedding_size, scope_name, weight_name, mod_size=None, offset=0):
fea_tensor = tf.convert_to_tensor([[1,2], [3,4], [5,6]])
tensor_weight = tf.convert_to_tensor([[0.2,0.5],[0.3,0.4],[0.2,0.3]])
tensor_length = tf.convert_to_tensor([[1], [2], [1]])
feature_size = 10
embedding_size = 4
scope_name = "dnn_preprocess"
weight_name = "doc_poi"

embed=preprocess_embedding(fea_tensor, feature_size,embedding_size, scope_name, weight_name)
w_embed = preprocess_weighted_avg_pooling_fixedsize(embed, tensor_weight, 2, "weighted")
sum_embed = preprocess_sum_pooling_fixedsize(embed, tensor_weight, tensor_length, "sum_pooling")

c = tf.convert_to_tensor(5)

d = a + 1

with tf.Session() as sess:
    print (a)
    print (b)
    print (c)
    print (sess.run(a))
    print (sess.run(d))
    print (sess.run(b))
    sess.run(tf.global_variables_initializer())
    print(sess.run(embed))
    print (sess.run(sum_embed))
    print (embed)
    print (sum_embed)
