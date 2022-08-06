import torch
import numpy as np

from utils import *

def ranklist(user_pos_test, topk_list):
    r = np.isin(topk_list, user_pos_test) + 0
    return r

def get_performance(user_pos_test, r):
    recall, ndcg = [], []

    for K in [20]:
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


def test_one_user(x):
    topk_list = x[0]
    u = x[1] # u_id
    user_pos_test = test_user_set[u]

    r = ranklist(user_pos_test, topk_list)
    output = get_performance(user_pos_test, r)

    return output


def test(args, model, user_dict, n_params):
    result = {'recall': np.zeros(1),
              'ndcg': np.zeros(1)}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    train_rating_matrix = user_dict['train_rating_matrix']

    u_batch_size = args.batch_size

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    if args.ssl == 1:
        item_gcn_emb, user_gcn_emb, _ = model.generate()
    else:
        item_gcn_emb, user_gcn_emb = model.generate()

    topk_lists, user_list = [], []

    batch_num = 1 if args.demo == 1 else n_user_batchs
    for u_batch_id in range(batch_num):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = trans_to_cuda(torch.LongTensor(np.array(user_list_batch)))
        u_g_embeddings = user_gcn_emb[user_batch]

        i_g_embddings = item_gcn_emb
        rate_batch = model.rating(u_g_embeddings, i_g_embddings.squeeze(1))
        for u in range(len(user_list_batch)):
            rate_batch[u][train_user_set[user_list_batch[u]]] = 0
        #rate_batch[train_rating_matrix[user_list_batch].toarray() > 0] = 0
        topk_list_batch = rate_batch.topk(20)[1].detach().cpu()

        for topk_list, u_id in zip(topk_list_batch, user_list_batch):
            topk_lists.append(topk_list)
            user_list.append(u_id)

    batch_result = []
    for rating_uid in zip(topk_lists, user_list):
        batch_result.append(test_one_user(rating_uid))

    for re in batch_result:
        result['recall'] += re['recall']/n_test_users
        result['ndcg'] += re['ndcg']/n_test_users

    return result
