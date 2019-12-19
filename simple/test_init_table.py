class InitTables(object):
    def __init__(self):
        self.table_feed_dict = dict()
        self.set_tables = [] 

    def add_table(self, weight_tensor, table_place, table):
        self.table_feed_dict[table_place] = table
        self.set_tables.append(weight_tensor.assign(table_place))

    def run(self, sess):
        if len(self.set_tables) > 0: 
            sess.run(self.set_tables, feed_dict = self.table_feed_dict)

init_tables = InitTables()

with tf.Session() as sess:
    init_tables.run(sess)
