import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CKER")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="sample", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    parser.add_argument("--demo", type=int, default=1, help="")

    parser.add_argument("--max_entity_num", type=int, default=16, help="")
    parser.add_argument("--max_item_num", type=int, default=16, help="")

    parser.add_argument("--mode", type=str, default="", help="")
    parser.add_argument("--ssl", type=int, default=0, help="")
    parser.add_argument("--alpha", type=float, default=0, help="")
    parser.add_argument("--scale", type=int, default=0, help="")

    return parser.parse_args()

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop