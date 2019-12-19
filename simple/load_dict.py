
def read_pretrain_table_file_use_statistics(table_file_name: str, dim: int, offset:int, key_size: int, table_size: int, index_dict:dict, weight_name:str) -> np.ndarray: # index_dict = {'key': value}
    import struct
    import mmh3
    #if table_file_name in pretrain_table_cache:
    #    return pretrain_table_cache[table_file_name]
    table = np.ndarray([table_size, dim], np.float32)
    valid = []
    if table_file_name.startswith("hdfs://"):
        tag = table_file_name.split("/")[-1]
        #local_file_name = str(mmh3.hash(table_file_name)) + '.table'
        local_file_name = tag + '.table'
        tf.logging.info(table_file_name)
        os.system(f'rm -rf {local_file_name} && hadoop fs -get {table_file_name} {local_file_name}')
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

