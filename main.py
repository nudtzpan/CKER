import random

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from data_loader import load_data
from CKER import Recommender
from evaluate import test
from utils import early_stopping, trans_to_cuda, parse_args
import pickle

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, batch_negs):

    feed_dict = {}
    entity_pairs = trans_to_cuda(train_entity_pairs[start:end])
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = trans_to_cuda(torch.LongTensor(batch_negs[start:end]))
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args
    args = parse_args()

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, multi_hops = load_data(args)

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    print ('load negative samples ...')
    all_negs = pickle.load(open(args.data_path + args.dataset + '/' + 'all_negs.pkl', 'rb'))

    """define model"""
    model = trans_to_cuda(Recommender(n_params, args, multi_hops))

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(500):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        batch_negs = all_negs[epoch][index]

        """training"""
        loss, s = 0, 0
        train_s_t = time()

        range_num = 2000 if args.demo == 1 else len(train_cf)
        while s + args.batch_size <= range_num:
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  batch_negs)
            batch_loss, _, _ = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        #if epoch % 10 == 9 or epoch == 1:
        if True:
            """testing"""
            test_s_t = time()
            ret = test(args, model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
