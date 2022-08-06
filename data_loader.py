import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import torch
from torch_sparse import SparseTensor

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))

def _si_norm_lap(adj):
    # D^{-1}A
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

def filter_triplets(triplets):
    triplet_num = len(triplets)

    kg_entity_set = defaultdict(list)
    kg_relation_set = defaultdict(list)
    for h, r, t in zip(triplets[:,0], triplets[:,1], triplets[:,2]):
        kg_entity_set[h].append(t)
        kg_relation_set[h].append(r)
    
    entity_ripples = []
    relation_ripples = []
    for e in range(0, n_entities):
        e_entities = kg_entity_set[e] if e in kg_entity_set.keys() else [e]
        e_relations = kg_relation_set[e] if e in kg_entity_set.keys() else [1]
        if len(list(set(e_entities))) < args.max_entity_num:
            this_e_ripples = e_entities
            this_r_ripples = e_relations
        else:
            e_indices = np.random.choice(list(set(e_entities)), size=args.max_entity_num, replace = False)
            this_e_ripples, this_r_ripples = [], []
            for i in range(len(e_entities)):
                if e_entities[i] in e_indices:
                    this_e_ripples.append(e_entities[i])
                    this_r_ripples.append(e_relations[i])
        entity_ripples.append(this_e_ripples)
        relation_ripples.append(this_r_ripples)

    triplets = []
    for h in range(0, n_entities):
        tails = entity_ripples[h]
        relations = relation_ripples[h]
        for r, t in zip(relations, tails):
            triplets.append([h, r, t])
    print ('Entity filter percentage = ', len(triplets)/triplet_num)

    return np.array(triplets)

def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    # consider two additional relations --- 'interact'.
    can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
    triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    triplets = filter_triplets(triplets)

    r_t_triplets = triplets[:,1:]

    """ get unique relation-tail pairs """
    unique_kg_pairs, index_pairs = np.unique(np.array(r_t_triplets), axis=0, return_inverse=True)

    unique_pair_num = unique_kg_pairs.shape[0]

    """ construct entity-unique_path mat """
    h_tripples = triplets[:,0]
    index_pairs = index_pairs
    vals = [1.] * len(h_tripples)
    kg_mat = sp.coo_matrix((vals, (h_tripples, index_pairs)), shape=(n_entities, unique_pair_num))
    kg_mat = _si_norm_lap(kg_mat)

    row = torch.LongTensor(np.array(kg_mat.row))
    col = torch.LongTensor(np.array(kg_mat.col))
    value = torch.Tensor(np.array(kg_mat.data))
    kg_mat = SparseTensor(row = row, col = col, value = value, sparse_sizes = (n_entities, unique_pair_num))

    return unique_kg_pairs, kg_mat


def build_graph(train_data):
    rd = []

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd.append([u_id, i_id])

    return rd

def filter_item_item_matrix(i_l, i_r):
    origin_num = len(i_l)
    train_item_set = defaultdict(list)
    for i, j in zip(i_l, i_r):
        train_item_set[i].append(j)

    item_ripples = []
    for i in range(n_items):
        i_items = train_item_set[i] if i in train_item_set.keys() else [i]
        if len(i_items) < args.max_item_num:
            item_ripples.append(i_items)
        else:
            # rank by frequency
            sort_list = sorted(Counter(i_items).items(), key=lambda x: x[1], reverse=True)
            sort_list_topk = list(np.array(sort_list)[:args.max_item_num,0])
            #i_samples = np.random.choice(i_items, size=args.max_item_num, replace = False)
            item_ripples.append(sort_list_topk)
    
    i_l, i_r = [], []
    for i in range(n_items):
        i_items = item_ripples[i]
        for j in i_items:
            i_l.append(i)
            i_r.append(j)
    
    print ('Item filter percentage = ', len(i_l)/origin_num)
    return i_l, i_r

def get_item_item_matrix(train_user_set):
    # TODO: to be optimized
    i_l, i_r = [], []
    for u in tqdm(train_user_set.keys(), ascii=True):
        items = train_user_set[u]
        for i in items:
            for j in items:
                if i != j:
                    i_l.append(i)
                    i_r.append(j)

    i_l, i_r = filter_item_item_matrix(i_l, i_r)
    
    vals = [1.] * len(i_l)
    ii_mat = sp.coo_matrix((vals, (i_l, i_r)), shape=(n_items, n_items))
    ii_mat = _si_norm_lap(ii_mat)

    row = torch.LongTensor(np.array(ii_mat.row))
    col = torch.LongTensor(np.array(ii_mat.col))
    value = torch.Tensor(np.array(ii_mat.data))
    ii_mat = SparseTensor(row = row, col = col, value = value, sparse_sizes = (n_items, n_items))

    return ii_mat

def build_sparse_relational_graph(relation_dict):
    print("Begin to build sparse relation matrix ...")
    np_mat = np.array(relation_dict)

    cf = np_mat.copy()
    vals = [1.] * len(cf)
    ui_mat = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_users, n_items))
    ui_mat = _si_norm_lap(ui_mat)

    row = torch.LongTensor(np.array(ui_mat.row))
    col = torch.LongTensor(np.array(ui_mat.col))
    value = torch.Tensor(np.array(ui_mat.data))
    ui_mat = SparseTensor(row = row, col = col, value = value, sparse_sizes = (n_users, n_items))

    return ui_mat


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('building the item-item mat ...')
    ii_mat = get_item_item_matrix(train_user_set)

    print('building the train rating mat ...')
    train_rating_matrix = sp.csr_matrix(([1.] * len(train_cf), (train_cf[:, 0], train_cf[:, 1])), shape=(n_users, n_items))

    print('combinating train_cf and kg data ...')
    kg_pairs, kg_mat = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    relation_dict = build_graph(train_cf)

    print('building the adj mat ...')
    ui_mat = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'train_rating_matrix': train_rating_matrix
    }

    return train_cf, test_cf, user_dict, n_params, [ui_mat, kg_mat, ii_mat, kg_pairs]

