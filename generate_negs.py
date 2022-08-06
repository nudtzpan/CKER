import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from data_loader import load_data
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="preprocess")
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="sample", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")
    parser.add_argument('--sample_num', type=int, default=100, help='number of epochs')
    return parser.parse_args()

def neg_sample(user_set):
    sample_from_items = list(all_items-set(user_set))
    user_negs = np.random.choice(sample_from_items, size=args.sample_num, replace=True) # 放回抽样
    #user_negs = random.sample(sample_from_items, args.sample_num) # 不放回抽样
    
    return user_negs

def sample_speed(train_cf_pairs, train_user_item_set):
    all_negs = []
    user_sets = [train_user_item_set[u] for u in train_cf_pairs[:,0]]
    for user_set in tqdm(user_sets, ascii=True):
        all_negs.append(neg_sample(user_set))
    all_negs = list(np.array(all_negs).T) # epoch * n_samples
    print ('all_negs.shape = ', np.array(all_negs).shape)
    pickle.dump(all_negs, open(args.data_path + args.dataset + '/' + 'all_negs.pkl', 'wb'))
    return all_negs

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

def load_data(args):
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')

    n_users = max(max(train_cf[:, 0]), max(test_cf[:, 0])) + 1
    n_items = max(max(train_cf[:, 1]), max(test_cf[:, 1])) + 1

    train_user_item_set = defaultdict(list)
    for u_id, i_id in train_cf:
        train_user_item_set[int(u_id)].append(int(i_id))

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items)
    }

    return train_cf, train_user_item_set, n_params

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)

    """read args"""
    args = parse_args()

    """build dataset"""
    train_cf, train_user_item_set, n_params = load_data(args)
    n_items = n_params['n_items']

    """cf data"""
    train_cf_pairs = np.array(train_cf)

    all_items = set(list(np.arange(n_items)))
    print('sample negs...')
    all_negs = sample_speed(train_cf_pairs, train_user_item_set)
