import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_users, n_items, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj

        self.emb_size = args.embed_size
        self.batch_size = args.batch_size

        self.node_dropout = eval(args.node_dropout)
        self.mess_dropout = eval(args.mess_dropout)

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        self.device = args.device

        self.embedding_dict, self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers

        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})
            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})
        return embedding_dict, weight_dict

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = X._indices()[:, dropout_mask]
        v = X._values()[dropout_mask] * (1 / keep_prob)
        return torch.sparse.FloatTensor(i, v, X.shape).to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        X = X.tocoo()
        i = torch.LongTensor([X.row, X.col])
        v = torch.FloatTensor(X.data)
        X = torch.sparse.FloatTensor(i, v, X.shape)
        return X

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), 1)
        neg_scores = torch.sum(torch.mul(users, neg_items), 1)

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2   # 源码中没有开方
        regularizer = regularizer / self.batch_size
        emb_loss = self.decay * regularizer

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        A_hat = self._dropout_sparse(self.sparse_norm_adj, 1-self.node_dropout[0], self.sparse_norm_adj._nnz()) \
                                    if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users:, :]

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
