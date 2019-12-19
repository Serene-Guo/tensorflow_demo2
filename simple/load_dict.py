import json
import tensorflow as tf
import numpy as np
import os

def load_field_idx_dict(dict_file):
    field_idx_dict = {}
    with open(dict_file) as f_dict:
        for line in f_dict:
            line = line.strip()
            line_arr = line.split('\t')
            if len(line_arr) != 3:
                continue
            field_name = line_arr[0]
            idx_dict_str = line_arr[2]
            try:
                idx_dict = json.loads(idx_dict_str, encoding='utf-8')
            except Exception as e:
                print (e)
                print (idx_dict_str)
                print (field_name)
                continue
            field_idx_dict[field_name] = idx_dict
    return field_idx_dict

#def read_pretrain_table_file_use_statistics(table_file_name: str, dim: int, offset:int, key_size: int, table_size: int, index_dict:dict, weight_name:str):
table_cache = {}
def read_pretrain_table_file_use_statistics(table_file_name, dim, offset, key_size, table_size, index_dict, weight_name):
    ### np.ndarray # index_dict = {'key': value}
    import struct
    #if table_file_name in pretrain_table_cache:
    #    return pretrain_table_cache[table_file_name]
    table = np.ndarray([table_size, dim], np.float32)
    print (table)
    hadoop_bin="/home/recsys/platform/hadoop-jd-2.7.3/bin/hadoop"
    valid = []
    if table_file_name.startswith("hdfs://"):
        tag = table_file_name.split("/")[-1]
        #local_file_name = str(mmh3.hash(table_file_name)) + '.table'
        local_file_name = tag + '.table'
        tf.logging.info(table_file_name)
        os.system(f'rm -rf {local_file_name} && {hadoop_bin} fs -get {table_file_name} {local_file_name}')
    else:
        local_file_name = table_file_name
    n_hit = 0
    found_idx_set = set()
    with open(local_file_name, 'rb') as bin_file:
        while True:
            key = bin_file.read(key_size)
            if len(key) < key_size:
                break
            key = key.strip(b'\x00').decode('utf-8')
            vector = bin_file.read(dim*4)
            vector = struct.unpack("{}f".format(dim), vector)
            idx = index_dict.get(key, -1)
            #if idx < 100:
            #tf.logging.info(type(key))
            #tf.logging.info(key + ":" + str(idx))
            ###  index_dict, value is always >=1. 
            if idx > 0 and idx < table_size and idx not in found_idx_set:
                found_idx_set.add(idx)
                table[idx] = vector
                n_hit += 1
            valid.append(idx)
    tf.logging.info(weight_name + " : " + str(n_hit) + "/" + str(len(index_dict)) + "=" + str(n_hit*1.0/len(index_dict)))
    for i in range(offset):
        table[i] = np.random.randn(dim)
    idxtable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(valid, valid), 0)
    table_cache[table_file_name] = (table, idxtable)
    return table, idxtable



if __name__ == "__main__":
    field_dict = load_field_idx_dict("./2121_dict")
    #print ( '2121_WordCount\t' +str(len(field_dict['2121_WordCount']))  + "\t" + json.dumps(field_dict['2121_WordCount'], ensure_ascii=False))
    table_file_name= "hdfs://hz-cluster9/user/datacenter/mlp/deep/fm_vec/20191214/cat"
    table_file_name ="hdfs://hz-cluster9/user/portal/ODM/RECOMMEND/datacenter/dnn_model/test_gff/fmvec/20191214/cat"
    table, idxtable = read_pretrain_table_file_use_statistics(table_file_name, 64, 1, 96, 231, field_dict['2121_WordCount'], 'u_cat2')
    print (table)
    print (idxtable)
    
    valid = [1,131,53,131,131]
    #idxtable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(valid, valid), 0)
    
    input_tensor = tf.constant([[0,1,2,131,53],[4,5,6,7,8]], dtype=tf.int32)
    not_valid_tensor = tf.constant(2, dtype=tf.int32)
    out = idxtable.lookup(input_tensor)
    out_0 = idxtable.lookup(not_valid_tensor)
    with tf.Session() as sess:
        idxtable.init.run()
        print (out)
        print(out.eval())  ### 1
        hit_rate = tf.divide(tf.count_nonzero(out, dtype=tf.int32), tf.size(out), name='test')
        print (hit_rate)
        print (sess.run(hit_rate))
        print (tf.count_nonzero(out, dtype=tf.int32))
        print (sess.run(tf.count_nonzero(out, dtype=tf.int32)))
        print (tf.size(out))
        print (sess.run(tf.size(out)))

