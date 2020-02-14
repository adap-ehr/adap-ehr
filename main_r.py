import sys
import numpy as np
from data_loader import WholeDataSet
import torch
from utils import Logger
import os
import os.path as osp
import trainer_r_r
import argparse


def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_2', default=0.002, type=float, help='initial learning rate for domain-specific part')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='sgd optimizer weight decay')
    parser.add_argument('--weight_decay_2', default=0.1, type=float, help='sgd optimizer weight decay for domain-specific part')
    parser.add_argument('--max_step', default=50, type=int, help='maximum step for training')
    parser.add_argument('--d5', default=16, type=int, help='dimension of domain-specific feature')
    parser.add_argument('--lambda_1', default=0.1, type=float, help='lambda_1')
    parser.add_argument('--lambda_2', default=0.01, type=float, help='lambda_2')
    parser.add_argument('--lambda_3', default=1.0, type=float, help='lambda_3')
    parser.add_argument('--num_train', default=300, type=int, help='number of training examples')
    parser.add_argument('--num_env', default=30, type=int, help='number of domains')

    parser.add_argument('--save_dir', default='./results', help='save directory for checkpoint')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    return parser.parse_args()


def generate_data(option):
    num_train = option.num_train
    num_env = option.num_env
    d5 = option.d5
    num_valid, num_test = 500, 500
    num_sample = num_train + num_valid + num_test
    d1, d2, d3, d4 = 32, 32, 32, 32
    b = np.random.uniform(-1.0, 1.0, (d3, 1))
    for idx in range(b.shape[0]):
        if b[idx] > 0:
            b[idx] += 0.5
        else:
            b[idx] -= 0.5
    b = b / np.sqrt(d3)

    c = np.random.uniform(-1.0, 1.0, (d4, 1))
    for idx in range(c.shape[0]):
        if c[idx] > 0:
            c[idx] += 0.5
        else:
            c[idx] -= 0.5
    c = c / np.sqrt(d4)

    train_X, train_X_s, train_X_d, train_y, train_d = [], [], [], [], []
    valid_X, valid_X_s, valid_X_d, valid_y, valid_d = [], [], [], [], []
    test_X, test_X_s, test_X_d, test_y, test_d = [], [], [], [], []
    for e in range(num_env):
        a_e = np.random.uniform(-1.0, 1.0, (d1, d2)) / np.sqrt(d1)

        d_e = np.random.uniform(-1.0, 1.0, (1, d5))
        for idx in range(d_e.shape[1]):
            if d_e[0, idx] > 0:
                d_e[0, idx] += 0.5
            else:
                d_e[0, idx] -= 0.5
        d_e = d_e / np.sqrt(d2)
        X1_e = np.random.randn(num_sample, d1)
        X2_e = np.matmul(X1_e, a_e) + np.random.randn(num_sample, d2)
        X3_e = X2_e + np.random.randn(num_sample, d3)
        X4_e = np.random.randn(num_sample, d4)

        E_Y_e = np.random.randn(num_sample, 1)
        Y_e = np.matmul(X3_e, b) + np.matmul(X4_e, c) + E_Y_e
        E_e = e * np.ones(num_sample).astype(np.int)

        if d5 > 0:
            E_X5_e = np.random.randn(num_sample, d5)
            X5_e = np.matmul(Y_e, d_e) + E_X5_e
            X_e = np.concatenate((X1_e, X2_e, X3_e, X4_e, X5_e), 1)
            train_X_d.append(X5_e[:num_train, :])
            valid_X_d.append(X5_e[num_train:num_train + num_valid, :])
            test_X_d.append(X5_e[num_train + num_valid:, :])
        else:
            X_e = np.concatenate((X1_e, X2_e, X3_e, X4_e), 1)
            train_X_d.append(X1_e[:num_train, :])
            valid_X_d.append(X1_e[num_train:num_train + num_valid, :])
            test_X_d.append(X1_e[num_train + num_valid:, :])
        train_X.append(X_e[:num_train, :])
        train_X_s.append(np.concatenate((X3_e, X4_e), 1)[:num_train, :])
        train_y.append(Y_e[:num_train, 0])
        train_d.append(E_e[:num_train])

        valid_X.append(X_e[num_train:num_train + num_valid, :])
        valid_X_s.append(np.concatenate((X3_e, X4_e), 1)[num_train:num_train + num_valid, :])
        valid_y.append(Y_e[num_train:num_train + num_valid, 0])
        valid_d.append(E_e[num_train:num_train + num_valid])

        test_X.append(X_e[num_train + num_valid:, :])
        test_X_s.append(np.concatenate((X3_e, X4_e), 1)[num_train + num_valid:, :])
        test_y.append(Y_e[num_train + num_valid:, 0])
        test_d.append(E_e[num_train + num_valid:])

    train_X, train_X_s, train_X_u, train_y, train_d = np.concatenate(train_X), np.concatenate(
        train_X_s), np.concatenate(train_X_d), np.concatenate(train_y), np.concatenate(train_d)
    valid_X, valid_X_s, valid_X_u, valid_y, valid_d = np.concatenate(valid_X), np.concatenate(
        valid_X_s), np.concatenate(valid_X_d), np.concatenate(valid_y), np.concatenate(valid_d)
    test_X, test_X_s, test_X_u, test_y, test_d = np.concatenate(test_X), np.concatenate(test_X_s), np.concatenate(
        test_X_d), np.concatenate(test_y), np.concatenate(test_d)
    return train_X, train_y, train_d, valid_X, valid_y, valid_d, test_X, test_y, test_d


def train(option):
    pre_fix = "syn_r/%d_%d_%d_lr#%f_lr2#%f_reg#%f_reg#%f" % (option.num_env,option.num_train,option.d5,option.lr,option.lr_2,option.weight_decay,option.weight_decay_2)
    result_dir = osp.join(option.save_dir,pre_fix)
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    logger = Logger(result_dir)

    all_mse_fs = []
    for s in range(0,1):
        np.random.seed(s)
        train_X, train_y, train_d, valid_X, valid_y, valid_d, test_X, test_y, test_d = generate_data(option)

        rnd_idx = np.array(range(len(train_y)))
        np.random.shuffle(rnd_idx)
        train_X,train_y,train_d = train_X[rnd_idx, :], train_y[rnd_idx], train_d[rnd_idx]
        print("Total number of features:", train_X.shape[1])

        train_loaders, valid_loaders, test_loaders = [], [], []
        for idx in range(0, test_d.max() + 1):
            train_X_d, train_y_d, train_d_d = train_X[train_d == idx, :], train_y[train_d == idx], train_d[train_d == idx]
            valid_X_d,  valid_y_d, valid_d_d = valid_X[valid_d == idx, :],  valid_y[valid_d == idx], valid_d[valid_d == idx]
            test_X_d,  test_y_d, test_d_d = test_X[test_d == idx, :], test_y[test_d == idx], test_d[test_d == idx]

            train_dataset = WholeDataSet(train_X_d.astype(np.float32), train_y_d.astype(np.float32), train_d_d)
            valid_dataset = WholeDataSet(valid_X_d.astype(np.float32), valid_y_d.astype(np.float32), valid_d_d)
            test_dataset = WholeDataSet(test_X_d.astype(np.float32), test_y_d.astype(np.float32), test_d_d)

            train_loaders.append(torch.utils.data.DataLoader(train_dataset,num_workers = 1,batch_size=option.batch_size,shuffle=True))
            valid_loaders.append(torch.utils.data.DataLoader(valid_dataset, num_workers = 1, batch_size=12800, shuffle=False))
            test_loaders.append(torch.utils.data.DataLoader(test_dataset, num_workers=1, batch_size=12800, shuffle=False))

        train_dataset_all = WholeDataSet(train_X.astype(np.float32), train_y.astype(np.float32), train_d)
        train_loader_all = torch.utils.data.DataLoader(train_dataset_all, num_workers=2,
                                                       batch_size=option.batch_size, shuffle=True)
        trainer_stable_r = trainer_r_r.Trainer(option,model_config={'Fs_paras': (train_X.shape[1], 32),
                                                                    'Fd_paras': (train_X.shape[1], 31),
                                                                    'Fc_paras': (train_X.shape[1], 32)},
                                                      logger=logger,num_domains=train_d.max() + 1)

        temp_mse = trainer_stable_r.train(train_loader_all, valid_loaders, test_loaders)
        all_mse_fs.append(temp_mse)

    logger.info(",".join([str(t_mse) for t_mse in all_mse_fs]))
    logger.info(str(np.array(all_mse_fs).mean()))
    logger.info(str(np.array(all_mse_fs).std()))
    logger.close()


if __name__ == '__main__':
    option = get_option()
    train(option)

