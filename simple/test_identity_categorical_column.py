import tensorflow as tf

pets = {'pets': [2,3,0,1], 'test': [-1,6,7,8]}  #猫0，狗1，兔子2，猪3

column = tf.feature_column.categorical_column_with_identity(
    key='pets',
    num_buckets=4)

column2 = tf.feature_column.categorical_column_with_identity(
    key='test',
    num_buckets=7,default_value=0)

indicator = tf.feature_column.indicator_column(column)
indicator2 = tf.feature_column.indicator_column(column2)

tensor = tf.feature_column.input_layer(pets, [indicator, indicator2])
#tensor = tf.feature_column.input_layer(pets, [column, column2])

with tf.Session() as session:
        print(session.run([tensor]))
