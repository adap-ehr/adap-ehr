# -*- coding: utf-8 -*-
import torch
from torch import nn
from layers import L0Mask,Extractor,Regressor, Reconstructor

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class FS_R(nn.Module):
    def __init__(self, option, model_config, num_domains):
        super(FS_R, self).__init__()
        self.S1 = L0Mask(model_config['input_dim'])
        self.S2 = L0Mask(model_config['input_dim'])

        self.C = Regressor(dims=(model_config['input_dim'], 1),if_bias = True)
        self.C_s_m = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = Regressor(dims=(model_config['input_dim'],1),if_bias = True).cuda()
            else:
                tempM = Regressor(dims=(model_config['input_dim'],1),if_bias = True)
            self.C_s_m.append(tempM)
        self.C_d_m = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = Regressor(dims=(model_config['input_dim'] * 2 + 1, 1), if_bias=True).cuda()
            else:
                tempM = Regressor(dims=(model_config['input_dim'] * 2 + 1, 1),if_bias = True)
            self.C_d_m.append(tempM)

    def forward_train(self, input_data, d):
        feat_sd, feat_c = self.S1(input_data)
        feat_s, feat_d = self.S2(feat_sd)
        output_s, output_s_m = self.C(feat_s),self.C_s_m[d](feat_s)
        output_d_m = self.C_d_m[d](torch.cat((feat_d,grad_reverse(feat_c,1.0),self.C(feat_s)),1))

        return feat_s, feat_d, output_s, output_s_m, output_d_m

    def forward_valid(self, input_data, d):
        feat_sd, feat_c = self.S1(input_data)
        feat_s, feat_d = self.S2(feat_sd)
        output_s, output_s_m = self.C(feat_s),self.C_s_m[d](feat_s)
        output_d_m = self.C_d_m[d](torch.cat((feat_d,0.0 * feat_c,self.C(feat_s)),1))

        return feat_s, feat_d, output_s, output_s_m, output_d_m

    def forward_train_s(self,input_data,d):
        feat_sd, feat_c = self.S1(input_data)
        feat_s, feat_d = self.S2(feat_sd)
        return feat_s, feat_d,self.C(feat_s),self.C_s_m[d](feat_s)

    def regularization(self):
        return self.S1.regularization()


class R_R(nn.Module):
    def __init__(self, option, model_config, num_domains):
        super(R_R, self).__init__()
        self.F_s = Extractor(model_config['Fs_paras'])
        self.F_d = Extractor(model_config['Fd_paras'])
        self.F_c = Extractor(model_config['Fc_paras'])

        self.R = Reconstructor([model_config['Fs_paras'][1] + model_config['Fd_paras'][1] + model_config['Fc_paras'][1],model_config['Fs_paras'][0]])
        self.C = Regressor(dims=(model_config['Fs_paras'][1], 1),if_bias = True)
        self.C_s_m = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = Regressor(dims=(model_config['Fs_paras'][1], 1), if_bias=True).cuda()
            else:
                tempM = Regressor(dims=(model_config['Fs_paras'][1], 1),if_bias = True)
            self.C_s_m.append(tempM)
        self.C_d_m = []
        for _ in range(num_domains):
            if option.cuda:
                tempM = Regressor(dims=(model_config['Fd_paras'][1] + model_config['Fc_paras'][1] + 1, 1), if_bias=True).cuda()
            else:
                tempM = Regressor(dims=(model_config['Fd_paras'][1] + model_config['Fc_paras'][1] + 1, 1),if_bias = True)
            self.C_d_m.append(tempM)

    def forward(self, input_data):
        feat_s, feat_d, feat_c = self.F_s(input_data), self.F_d(input_data), self.F_c(input_data)
        output_s = self.C(feat_s)
        outputs_s_m,outputs_d_m = [],[]
        for d_idx,tempM in enumerate(self.C_s_m):
            outputs_s_m.append(tempM(feat_s))

        for tempM in self.C_d_m:
            outputs_d_m.append(tempM(torch.cat((feat_d,grad_reverse(feat_c,1.0),self.C(feat_s)),1)))

        return feat_s, feat_d, output_s, outputs_s_m, outputs_d_m

    def forward_re(self,input_data):
        feat_s, feat_d, feat_c = self.F_s(input_data), self.F_d(input_data), self.F_c(input_data)
        reconstruct_x = self.R(torch.cat([feat_s, feat_d, feat_c], 1))

        return reconstruct_x

    def forward_s(self, input_data):
        feat_s = self.F_s(input_data)
        output_s = self.C(feat_s)
        outputs_s_m = []
        for (d_idx,tempM) in enumerate(self.C_s_m):
            outputs_s_m.append(tempM(feat_s))
        return output_s,outputs_s_m


