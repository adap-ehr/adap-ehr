import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import nn

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0Mask(Module):
    """Implementation of L0 regularization for the input units of a feature selection layer, modified from the code of l0 full connected layers in https://github.com/AMLab-Amsterdam/L0_regularization"""
    def __init__(self, in_features, droprate_init=0.1, temperature=2./3., local_rep=False, **kwargs):
        super(L0Mask, self).__init__()
        self.in_features = in_features
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw = torch.mean(1 - self.cdf_qz(0))
        return logpw

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def forward(self, input):
        z = self.sample_z(input.size(0), sample=self.training)
        xin = input.mul(z)
        xin_r = input.mul(1.0 - z)
        return xin,xin_r

    def sample_mask(self):
        pi = torch.sigmoid(self.qz_loga).view(1, self.in_features)
        return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def __repr__(self):
        s = ('{name}({in_features}, droprate_init={droprate_init}, '
             'temperature={temperature}, '
             'local_rep={local_rep}')
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Extractor(nn.Module):
    """Feature extractor in representation learning model"""
    def __init__(self,dims = (32,32)):
        super(Extractor,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        self.fc1.weight.data.normal_(0, 1.0 / self.fc1.in_features)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        return self.fc1(x)


class Reconstructor(nn.Module):
    """Reconstructor in representation learning model"""
    def __init__(self,dims = (32,1),if_bias = True):
        super(Reconstructor,self).__init__()
        self.fc1 = nn.Linear(dims[0],dims[1])
        nn.init.constant_(self.fc1.weight, 0)
        if if_bias:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x):
        return self.fc1(x)


class Regressor(nn.Module):
    def __init__(self,dims = (32,1),if_bias = True):
        super(Regressor,self).__init__()
        self.fc1 = nn.Linear(dims[0],1,bias=if_bias)
        nn.init.constant_(self.fc1.weight, 0)
        if if_bias:
            nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x, bias = None):
        out = self.fc1(x)
        return out
