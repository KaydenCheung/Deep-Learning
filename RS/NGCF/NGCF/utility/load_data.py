import numpy as np
import random as rd
import scipy.sparse as sp
from time import time


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = self.path + '/train.txt'
        test_file = self.path + '/test.txt'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        self.n_users += 1
        self.n_items += 1

        self.print_statistics()

        # 格式:(row, col) value
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # 基于字典的稀疏矩阵，适合用来添加元素，然后转换成其他运算的格式

        self.train_items, self.test_set = {}, {}
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]]

                    for i in items:
                        self.R[uid, i] = 1

                    self.train_items[uid] = items

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    items = [int(i) for i in l[1:]]
                    self.test_set[uid] = items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users+self.n_items, self.n_users+self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):      # D^(-1/2) * A * D^(-1/2)
            rowsum = np.array(adj.sum(1))
            D = np.power(rowsum, -0.5).flatten()
            D[np.isinf(D)] = 0
            D = sp.diags(D)
            bi_lap = D.dot(adj).dot(D)
            return bi_lap.tocoo()

        def mean_adj_single(adj):            # D^(-1) * A
            rowsum = np.array(adj.sum(1))
            D = np.power(rowsum, -1).flatten()
            D[np.isinf(D)] = 0
            D = sp.diags(D)
            norm_adj = D.dot(adj)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(self.n_users + self.n_items))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_batch = []
            pos_items = self.train_items[u]
            while True:
                if len(neg_batch) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in pos_items and neg_id not in neg_batch:
                    neg_batch.append(neg_id)
            return neg_batch

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))


# d = Data('../data/gowalla', 128)
# adj_mat, norm_adj_mat, mean_adj_mat = d.get_adj_mat()
