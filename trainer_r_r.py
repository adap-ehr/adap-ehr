# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models
import numpy as np
from torch.autograd import grad

class Trainer(object):
    def __init__(self, option, model_config, logger, num_domains=2):
        self.option = option
        self.num_domains = num_domains
        self.lambda_1 = option.lambda_1
        self.lambda_2 = option.lambda_2
        self.lambda_3 = option.lambda_3
        self.logger = logger
        self._build_model(model_config, num_domains)
        self._set_optimizer()

    def _build_model(self, model_config, num_domains=2):
        self.net = models.R_R(self.option, model_config, num_domains)
        self.loss = nn.MSELoss()

        if self.option.cuda:
            self.net.cuda()
            self.loss.cuda()

    def _set_optimizer(self):
        self.optim_fs = optim.Adam(self.net.F_s.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_fd = optim.Adam(self.net.F_d.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_fc = optim.Adam(self.net.F_c.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_r = optim.Adam(self.net.R.parameters(), lr=self.option.lr, weight_decay=0.0)
        self.optim_c = optim.Adam(self.net.C.parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay)
        self.optim_c_s_m = []
        for idx in range(len(self.net.C_s_m)):
            self.optim_c_s_m.append(optim.Adam(self.net.C_s_m[idx].parameters(), lr=self.option.lr,
                                              weight_decay=self.option.weight_decay_2))
        self.optim_c_d_m = []
        for idx in range(len(self.net.C_d_m)):
            self.optim_c_d_m.append(
                optim.Adam(self.net.C_d_m[idx].parameters(), lr=self.option.lr, weight_decay=self.option.weight_decay_2))

    def _reset_grad(self):
        self.optim_fs.zero_grad()
        self.optim_fd.zero_grad()
        self.optim_fc.zero_grad()
        self.optim_r.zero_grad()
        self.optim_c.zero_grad()
        for idx in range(len(self.optim_c_s_m)):
            self.optim_c_s_m[idx].zero_grad()
        for idx in range(len(self.optim_c_d_m)):
            self.optim_c_d_m[idx].zero_grad()

    def _group_step(self, optim_list):
        for i in range(len(optim_list)):
            optim_list[i].step()

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            for idx in range(len(self.net.C_s_m)):
                self.net.C_s_m[idx].train()
            for idx in range(len(self.net.C_s_m)):
                self.net.C_s_m[idx].train()
        else:
            self.net.eval()
            for idx in range(len(self.net.C_d_m)):
                self.net.C_d_m[idx].eval()
            for idx in range(len(self.net.C_d_m)):
                self.net.C_d_m[idx].eval()

    def _create_masked_output(self,output_mul, t_d):
        output_f = torch.zeros_like(output_mul[0])
        for j, t_output in enumerate(output_mul):
            mask = t_d == j
            output_f += t_output * mask.float()
        return output_f

    # @profile
    def _train_step(self, data_loader):
        self._mode_setting(is_train=True)
        for i, (x, y, d) in enumerate(data_loader):
            x = self._get_variable(x)
            y = self._get_variable(y)
            d = self._get_variable(d)

            self._reset_grad()

            pred_y_s, pred_ys_s_m = self.net.forward_s(x)
            pred_y_s_m = self._create_masked_output(pred_ys_s_m, d)
            loss_pred_y_s = self.loss(pred_y_s[:,0], y[:, 0])
            loss_pred_y_s_m = self.loss(pred_y_s_m[:,0], y[:, 0])
            loss = 1.0 * loss_pred_y_s + 1.0 * loss_pred_y_s_m
            loss.backward()
            self._group_step([self.optim_c] + self.optim_c_s_m)
            self._reset_grad()

            pred_y_s, pred_ys_s_m = self.net.forward_s(x)
            pred_y_s_m = self._create_masked_output(pred_ys_s_m, d)
            loss_disc = ((pred_y_s - pred_y_s_m) ** 2).mean()
            loss_pred_y_s = self.loss(pred_y_s[:,0], y[:, 0])
            loss = 1.0 * loss_pred_y_s + self.lambda_1 * loss_disc
            loss.backward()
            self._group_step([self.optim_fs])
            self._reset_grad()

            pred_y_s, pred_ys_s_m = self.net.forward_s(x)
            loss_pred_y_s = self.loss(pred_y_s[:, 0], y[:, 0])
            g = grad(loss_pred_y_s, self.net.C.fc1.weight)[0]
            loss_grad = self.lambda_2 * g.norm(2)
            loss_grad.requires_grad_()
            loss_grad.backward()
            self._group_step([self.optim_fs])
            self._reset_grad()

            feat_s, feat_d, pred_y_s, pred_ys_s_m, pred_ys_d_m = self.net(x)
            pred_y_d_m = self._create_masked_output(pred_ys_d_m, d)
            loss_pred_y_d_m = self.loss(pred_y_d_m[:,0], y[:, 0])

            norm_d = torch.sqrt((feat_d ** 2).sum(axis=0)).unsqueeze(0)
            norm_s = torch.sqrt((feat_s ** 2).sum(axis=0)).unsqueeze(-1)
            ip_ds = torch.tensordot(torch.transpose(feat_s, 1, 0), feat_d, 1),
            loss_ortho = torch.abs(ip_ds[0] / torch.tensordot(norm_s,norm_d,1)).mean()

            loss = loss_pred_y_d_m + self.lambda_3 * loss_ortho
            loss.backward()
            self._group_step([self.optim_fd, self.optim_fc] + self.optim_c_d_m)
            self._reset_grad()

            r_x = self.net.forward_re(x)
            loss_re = ((x - r_x) ** 2).mean()
            loss_re.backward()
            self._group_step([self.optim_fs, self.optim_fd, self.optim_fc, self.optim_r])
            self._reset_grad()

    def _valid(self, data_loaders, step, valid_flag):
        self._mode_setting(is_train=False)
        msg = "Step: %d" % step
        self.logger.info(msg)

        if valid_flag == 0:
            valid_str = "Train"
        elif valid_flag == 1:
            valid_str = "Valid"
        else:
            valid_str = "test"
        msg = "[VALID-%s]  (epoch %d)" % (valid_str, step)
        self.logger.info(msg)
        for (d_idx,data_loader) in enumerate(data_loaders):
            all_pred_y_s,all_pred_y_s_m, all_pred_y_d_m,all_labels = [],[],[],[]
            for i, (x, y, d) in enumerate(data_loader):
                x = self._get_variable(x)
                y = self._get_variable(y)
                d = self._get_variable(d)

                feat_s, feat_d, pred_y_s, pred_ys_s_m, pred_ys_d_m = self.net(x)

                pred_y_s_m = self._create_masked_output(pred_ys_s_m, d)
                pred_y_d_m = self._create_masked_output(pred_ys_d_m, d)

                all_pred_y_s_m.append(self._get_numpy_from_variable(pred_y_s_m))
                all_pred_y_s.append(self._get_numpy_from_variable(pred_y_s))
                all_pred_y_d_m.append(self._get_numpy_from_variable(pred_y_d_m))
                all_labels.append(self._get_numpy_from_variable(y))

        all_pred_y_s = np.concatenate(all_pred_y_s, 0)
        all_pred_y_s_m = np.concatenate(all_pred_y_s_m, 0)
        all_pred_y_d_m = np.concatenate(all_pred_y_d_m, 0)
        all_labels = np.concatenate(all_labels, 0)
        mse_s, mse_s_m = ((all_labels - all_pred_y_s) ** 2).mean(),((all_labels - all_pred_y_s_m) ** 2).mean()
        mse_d_m = ((all_labels - all_pred_y_d_m) ** 2).mean()

        msg = "Average Mse-S: %.3f  S_M: %.3f D_M: %.3f" % (mse_s,mse_s_m,mse_d_m)
        self.logger.info(msg)
        return np.array(mse_d_m).mean()

    def train(self, train_loader, val_loaders=None, test_loaders=None):
        self._mode_setting(is_train=True)
        best_valid_mse_d_m, best_test_mse_d_m = 9999.0, 9999.0
        best_step_d_m,stop_cnt = -1, 0
        for step in range(self.option.max_step):
            self._train_step(train_loader)
            self._valid([train_loader], step, 0)
            valid_mse_d_m = self._valid(val_loaders, step, 1)
            test_mse_d_m = self._valid(test_loaders, step, 2)
            if valid_mse_d_m < best_valid_mse_d_m:
                best_valid_mse_d_m = valid_mse_d_m
                best_test_mse_d_m = test_mse_d_m
                best_step_d_m = step
                stop_cnt = 0
            else:
                stop_cnt += 1
            msg = "[VALID-BEST-P]: %.3f, %.3f, %d, %d" % (best_valid_mse_d_m, best_test_mse_d_m, best_step_d_m, stop_cnt)
            self.logger.info(msg)
            if stop_cnt >= 5:
                break
        return best_test_mse_d_m

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

    def _get_numpy_from_variable(self, input):
        if self.option.cuda:
            return input.data.cpu().numpy()
        return input.data.numpy()
