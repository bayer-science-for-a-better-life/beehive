import shutil
import pickle
import lmdb


class DBConnector(object):

    map_size = int(4e9)  # 4GB
    max_dbs = 3  # train, val, test

    def __init__(self, db_name):
        self.db_name = db_name

    def open(self):
        self.db = lmdb.open(self.db_name, max_dbs=self.max_dbs, map_size=self.map_size)

    def close(self):
        self.db.close()

    def clear_db(self):
        try:
            shutil.rmtree(self.db_name)
        except FileNotFoundError:
            pass

    def write_data(self, key, value, dataset):
        """Write value into db as (features, target), use int key"""
        sub_db = self.db.open_db(dataset.encode())
        with self.db.begin(write=True) as txn:
            cur = txn.cursor(sub_db)
            cur.put(DBConnector._int2key(key), pickle.dumps(value))

    def get_values(self, key_list, dataset):
        """Get records by int keys as tuple of (features, targets)"""
        sub_db = self.db.open_db(dataset.encode())
        with self.db.begin() as txn:
            cur = txn.cursor(sub_db)
            targets = []
            feats = []
            for key in key_list:
                value = pickle.loads(cur.get(DBConnector._int2key(key)))
                feats.append(value[0])
                targets.append(value[1])
        return feats, targets

    def get_all_targets(self, dataset):
        """Iterate over all db entries to get list of targets"""
        sub_db = self.db.open_db(dataset.encode())
        targets = []
        with self.db.begin() as txn:
            cur = txn.cursor(sub_db)
            cur.first()
            for k, v in cur:
                targets.append(pickle.loads(v)[1])
        return targets

    def get_all_features(self, dataset):
        """Iterate over all db entries to get list of features"""
        sub_db = self.db.open_db(dataset.encode())
        with self.db.begin() as txn:
            cur = txn.cursor(sub_db)
            cur.first()
            feats = []
            for k, v in cur:
                feats.append(pickle.loads(v)[0])
        return feats

    def get_next_key(self, dataset):
        sub_db = self.db.open_db(dataset.encode())
        with self.db.begin() as txn:
            cur = txn.cursor(sub_db)
            if cur.last():
                return DBConnector._key2int(cur.key()) + 1
            else:
                return 0

    @staticmethod
    def _int2key(num):
        return chr(int(num)).encode()

    @staticmethod
    def _key2int(key):
        return ord(key.decode())
