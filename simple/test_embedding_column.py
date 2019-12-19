import tensorflow as tf

from tensorflow import feature_column as feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder
import sys
#sys.path.insert(0, "./tf_example/")
import base64
#from tf_example import example_pb2
#import  tensorflow.train.example as example
from tensorflow.core.example import example_pb2
#from tensorflow.core.example import feature_pb2


def test_embedding():
    color_data = {'color': [['G'], ['B'], ['B'], ['R']]}  # 4行样本
 
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
   
    color_embeding = feature_column.embedding_column(color_column, 7)
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
    builder = _LazyBuilder(color_data)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print(color_column_tensor.weight_tensor)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('embeding' + '-' * 40)
        print(session.run([color_embeding_dense_tensor]))

def test_weighted_categorical_feature_embedding():
    color_data = {'color': [['R','R'], ['G','G'], ['B','B'], ['G','R'], ['G', 'B'], ['B','R']],
                   'weight': [[0.5,0.5], [0.5,0.5], [0.5,0.5], [0.3,0.2], [0.4, 0.3], [0.4,0.6]]}  # 6行样本
 
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
  	
    color_embeding = feature_column.embedding_column(color_column, 7, combiner="sum")
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
    
    color_weight_categorical_column = feature_column.weighted_categorical_column(color_column, 'weight')
    color_embeding_weighted = feature_column.embedding_column(color_weight_categorical_column, 7, combiner="sum")
    color_embeding_dense_tensor_2 = feature_column.input_layer(color_data, [color_embeding_weighted])
    
    builder = _LazyBuilder(color_data)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_weighted_tensor = color_weight_categorical_column._get_sparse_tensors(builder) ## is a pair (id_tensor, weight_tensor)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print ("color column weight:")
        print(color_column_tensor.weight_tensor) 
        print ("color column weighted categorical,  weight:")
        print(session.run([color_weighted_tensor.id_tensor]))
        print(session.run([color_weighted_tensor.weight_tensor]))
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('embeding' + '-' * 40)
        print(session.run([color_embeding_dense_tensor]))
        print('embeding weighted categorical column')
        print(session.run([color_embeding_dense_tensor_2]))


def test_multi_value_embedding():
    color_data = {'color': [['G','G'], ['G','B'], ['B','B'], ['G','R'], ['R', 'R'], ['B','R']]}
 
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
   
    color_embeding = feature_column.embedding_column(color_column, 7)
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
    builder = _LazyBuilder(color_data)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
 
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('embeding' + '-' * 40)
        print(session.run([color_embeding_dense_tensor]))

def test_weighted_categorical_column():
    f_in = open("new.tf.rec.base64", "r")
    for line in f_in:
        try:
            b = base64.b64decode(line.strip())
        except Exception as e:
            sys.stderr.write(e)
            continue
        
        exa = example_pb2.Example()
        print ("before parse proto...........")
        try:
           exa.ParseFromString(b)
        except Exception as e:
            sys.stderr.write(e.str())
            continue
        print ("after parse proto........")
        #print (exa) 
        u_pocs_l1_norm = feature_column.categorical_column_with_hash_bucket("u_pocs_l1_norm", 3000)
        u_pocs_l1_norm_weighted =  feature_column.weighted_categorical_column(u_pocs_l1_norm, weight_feature_key='u_pocs_l1_norm_val')
        feature_columns = [u_pocs_l1_norm_weighted]
        features = tf.parse_single_example(b, tf.feature_column.make_parse_example_spec(feature_columns))
        print (features["u_pocs_l1_norm"])
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            print(session.run(features["u_pocs_l1_norm"]))
        break

def test_identity_categorical_column():
    pass

if __name__ == "__main__":
	#test_embedding()
    #test_multi_value_embedding()
    #test_weighted_categorical_feature_embedding()
    test_weighted_categorical_column()
