import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import trans_to_cuda
from torch_sparse import matmul

class GraphConv(nn.Module):
    def __init__(self, emb_dim, n_hops, nums, n_relations, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_hops = n_hops
        self.nums = nums
        self.n_items = self.nums[1]
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, emb_dim))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, emb_dim]
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def item_repre_learning(self, entity_emb, item_emb, multi_hops):
        kg_mat, ii_mat, kg_pairs = multi_hops[1], multi_hops[2], multi_hops[3]
        """kg aggregate"""
        unique_neigh = entity_emb[kg_pairs[:,1]] * self.weight[kg_pairs[:,0]-1]
        entity_agg = matmul(kg_mat, unique_neigh)
        """ii aggregate """
        item_agg = matmul(ii_mat, item_emb)

        return entity_agg, item_agg

    def forward(self, entity_emb, item_emb, multi_hops, mess_dropout=True):
        entity_emb_multi, item_emb_multi = [], []
        entity_emb_multi.append(entity_emb)
        item_emb_multi.append(item_emb)
        for i in range(self.n_hops):
            entity_emb, item_agg = self.item_repre_learning(entity_emb, item_emb, multi_hops)

            """message dropout"""
            #if mess_dropout:
            #    entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            item_agg = F.normalize(item_agg)

            """result emb"""
            entity_emb_multi.append(entity_emb)
            item_emb_multi.append(item_agg)

        return entity_emb_multi, item_emb_multi


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, multi_hops):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.nums = [self.n_users, self.n_items, self.n_entities]

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.mode = args_config.mode
        self.ssl = args_config.ssl
        self.scale = args_config.scale
        self.alpha = args_config.alpha

        self.multi_hops = multi_hops

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

        # [n_users, n_entities]
        self.ui_mat = trans_to_cuda(self.multi_hops[0]) # ui_mat
        self.multi_hops[1] = trans_to_cuda(self.multi_hops[1]) # kg_mat
        self.multi_hops[2] = trans_to_cuda(self.multi_hops[2]) # ii_mat

    def _init_model(self):
        return GraphConv(emb_dim=self.emb_size,
                         n_hops=self.context_hops,
                         nums=self.nums,
                         n_relations=self.n_relations,
                         mess_dropout_rate=self.mess_dropout_rate)
    
    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        if self.ssl == 1:
            item_gcn_emb, user_gcn_emb, [user_gcn_emb_kg, user_gcn_emb_ii] = self.generate()
        else:
            item_gcn_emb, user_gcn_emb = self.generate()
        u_e = user_gcn_emb[user]
        if self.ssl == 1:
            u_e_kg = user_gcn_emb_kg[user]
            u_e_ii = user_gcn_emb_ii[user]
            ssl_loss = self.create_ssl_loss(u_e_kg, u_e_ii)
        else:
            ssl_loss = 0
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e, ssl_loss)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        item_emb = entity_emb[:self.n_items]
        entity_kg_multi, item_ii_multi = self.gcn(entity_emb, item_emb, self.multi_hops, mess_dropout=False)
        item_kg_multi = []
        for i in range(len(entity_kg_multi)):
            item_kg_multi.append(entity_kg_multi[i][:self.n_items])

        """ fuse item infor of both kg and ii """
        item_fuse_multi = []
        for i in range(self.context_hops+1):
            if self.mode == 'kg':
                item_fuse_multi.append(item_kg_multi[i])
            elif self.mode == 'ii':
                item_fuse_multi.append(item_ii_multi[i])
            elif self.mode == 'fuse':
                item_fuse_multi.append(item_kg_multi[i] + item_ii_multi[i])
        
        """ generate final item representation """
        item_gcn_emb = item_fuse_multi[0]
        for i in range(1, self.context_hops+1):
            item_gcn_emb = torch.add(item_gcn_emb, item_fuse_multi[i])

        """ generate final user representation from kg """
        user_gcn_emb_kg = user_emb
        for i in range(1, self.context_hops+1):
            user_gcn_emb_kg_i = matmul(self.ui_mat, item_kg_multi[i-1])
            user_gcn_emb_kg = torch.add(user_gcn_emb_kg, F.normalize(user_gcn_emb_kg_i))

        """ generate final user representation from ii """
        user_gcn_emb_ii = user_emb
        for i in range(1, self.context_hops+1):
            user_gcn_emb_ii_i = matmul(self.ui_mat, item_ii_multi[i-1])
            user_gcn_emb_ii = torch.add(user_gcn_emb_ii, F.normalize(user_gcn_emb_ii_i))

        """ fuse item infor of both kg and ii """
        if self.mode == 'kg':
            user_gcn_emb = user_gcn_emb_kg
        elif self.mode == 'ii':
            user_gcn_emb = user_gcn_emb_ii
        elif self.mode == 'fuse':
            user_gcn_emb = user_gcn_emb_kg + user_gcn_emb_ii

        if self.ssl == 1:
            return item_gcn_emb, user_gcn_emb, [user_gcn_emb_kg, user_gcn_emb_ii]
        else:
            return item_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def l2_norm(self, x):
        y = x / torch.sqrt(torch.sum(x**2, -1, keepdim=True) + 1e-24)
        return y

    def create_ssl_loss(self, user_emb_kg, user_emb_ii):
        scores = torch.matmul(user_emb_kg, user_emb_ii.T)
        bs = user_emb_kg.shape[0]

        if self.scale != 0:
            user_emb_kg_norm, user_emb_ii_norm = self.l2_norm(user_emb_kg), self.l2_norm(user_emb_ii)
        scores = torch.matmul(user_emb_kg_norm, user_emb_ii_norm.T) # bs * bs
        if self.scale != 0:
            scores *= self.scale
        scores = torch.log_softmax(scores, -1)
        loss_temp = scores[torch.arange(bs).long(), torch.arange(bs).long()] # bs

        return -1 * torch.mean(loss_temp)

    def create_bpr_loss(self, users, pos_items, neg_items, ssl_loss):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        if self.ssl == 1:
            return mf_loss + emb_loss + self.alpha*ssl_loss, mf_loss, emb_loss
        else:
            return mf_loss + emb_loss, mf_loss, emb_loss
