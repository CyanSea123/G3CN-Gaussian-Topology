import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class G3CN(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(G3CN, self).__init__()
        self.n_head=8
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels//rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv0 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv21=nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv23 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv31 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        self.softmax=nn.Softmax(-1)
        self.sigmoid=nn.Sigmoid()
        self.E=1-torch.eye(25)
        D=np.array([[1,1,3,4,3,4,5,6,3,4,5,6,1,2,3,4,1,2,3,4,2,7,7,7,7],
               [1,1,2,3,2,3,4,5,2,3,4,5,2,3,4,5,2,3,4,5,1,6,6,6,6],
               [3,2,1,1,2,3,4,5,2,3,4,5,4,5,6,7,4,5,6,7,1,6,6,6,6],
               [4,3,1,1,3,4,5,6,3,4,5,6,5,6,7,8,5,6,7,8,2,7,7,7,7],
               [3,2,2,3,1,1,2,3,2,3,4,5,4,5,6,7,4,5,6,7,1,4,4,6,6],
               [4,3,3,4,1,1,1,2,3,4,5,6,5,6,7,8,5,6,7,8,2,3,3,7,7],
               [5,4,4,5,2,1,1,1,4,5,6,7,6,7,8,9,6,7,8,9,3,2,2,8,8],
               [6,5,5,6,3,2,1,1,5,6,7,8,7,8,9,10,7,8,9,10,4,1,1,9,9],
               [3,2,2,3,2,3,4,5,1,1,2,3,4,5,6,7,4,5,6,7,1,6,6,4,4],
               [4,3,3,4,3,4,5,6,1,1,1,2,5,6,7,8,5,6,7,8,2,7,7,3,3],
               [5,4,4,5,4,5,6,7,2,1,1,1,6,7,8,9,6,7,8,9,3,8,8,2,2],
               [6,5,5,6,5,6,7,8,3,2,1,1,7,8,9,10,7,8,9,10,4,9,9,1,1],
               [1,2,4,5,4,5,6,7,4,5,6,7,1,1,2,3,2,3,4,5,3,8,8,8,8],
               [2,3,5,6,5,6,7,8,5,6,7,8,1,1,1,2,3,4,5,6,4,9,9,9,9],
               [3,4,6,7,6,7,8,9,6,7,8,9,2,1,1,1,4,5,6,7,5,10,10,10,10],
               [4,5,7,8,7,8,9,10,7,8,9,10,3,2,1,1,5,6,7,8,6,11,11,11,11],
               [1,2,4,5,4,5,6,7,4,5,6,7,2,3,4,6,1,1,2,3,3,8,8,8,8],
               [2,3,5,6,5,6,7,8,5,6,7,8,3,4,5,6,1,1,1,2,4,9,9,9,9],
               [3,4,6,7,6,7,8,9,6,7,8,9,4,5,6,7,2,1,1,1,5,10,10,10,10],
               [4,5,7,8,7,8,9,10,7,8,9,10,5,6,7,8,3,2,1,1,6,11,11,11,11],
               [2,1,1,2,1,2,3,4,1,2,3,4,3,4,5,6,3,4,5,6,1,5,5,5,5],
               [7,6,6,7,4,3,2,1,6,7,8,9,8,9,10,11,8,9,10,11,5,1,2,10,10],
               [7,6,6,7,4,3,2,1,6,7,8,9,8,9,10,11,8,9,10,11,5,2,1,10,10],
               [7,6,6,7,6,7,8,9,4,3,2,1,8,9,10,11,8,9,10,11,5,10,10,1,2],
               [7,6,6,7,6,7,8,9,4,3,2,1,8,9,10,11,8,9,10,11,5,10,10,2,1]])-1
        self.D=Variable(torch.from_numpy(D.astype(np.float32)), requires_grad=False)


    def forward(self,A1, x, A=None, alpha=1, get_topology=False, label=None, mix_top=False):
        E = self.E.cuda(x.get_device())
        D=self.D.cuda(x.get_device())
        x1, x2,x3,x4,x5= self.conv1(x).mean(-2), self.conv2(x).mean(-2),self.conv0(x),self.conv13(x).mean(-2), self.conv23(x).mean(-2)
        mm=self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        mm1=self.tanh(x4.unsqueeze(-1) - x5.unsqueeze(-2))
        mm1=mm1/(torch.max(torch.abs(mm1),dim=-1,keepdim=True)[0]+ 1e-12)
        score=torch.einsum('ncuv,ncvl->ncul', mm1, (torch.exp(-D*D))[None,None,:,:])
        

        x_sigma=self.conv21(mm*score)


        x_t = torch.einsum('ncuv,nctv->nctu', x_sigma*E[None,None,:,:], x3)
        h_t_o=x3

        x_wr=self.conv3(x_t)
        h_wr = self.conv4(h_t_o)
        x_r = self.sigmoid(x_wr+h_wr)

        x_wz = self.conv5(x_t)
        h_wz = self.conv6(h_t_o)
        x_z = self.sigmoid(x_wz + h_wz)

        x_t_o=self.tanh(self.conv7(x_r*h_t_o)+self.conv8(x_t))
        x_out= x_t_o*x_z +h_t_o*(1-x_z)

        x_out +=  torch.einsum('ncuv,nctv->nctu', A.unsqueeze(0).unsqueeze(0) if A is not None else 0, x3)


        return x_out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(3):
            self.convs.append(G3CN(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, get_topology=False, label=None, mix_top=False):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        topology = []
        A1 = self.A.cuda(x.get_device())
        for i in range(3):
            if get_topology:
                z, top = self.convs[i](x, A[i], self.alpha, True, label=label, mix_top=mix_top)
                y = z + y if y is not None else z
                topology.append(top)
            else:
                z = self.convs[i](A1,x, A[i], self.alpha, label=label, mix_top=mix_top)
                y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        if get_topology:
            return y, torch.cat(topology, dim=1).mean(1, keepdim=True)
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:channel//2])
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2], use_pe=False):
        super(TCN_GCN_unit, self).__init__()
        self.use_pe = use_pe
        if self.use_pe:
            self.pos_enc = PositionalEncoding(in_channels, A.shape[1], 64, 'spatial')

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, get_topology=False, label=None, mix_top=False):
        if self.use_pe:
            x = self.pos_enc(x)
        if get_topology:
            tmp, top = self.gcn1(x, True, label=label, mix_top=mix_top)
            y = self.relu(self.tcn1(tmp) + self.residual(x))
            return y, top
        else:
            y = self.relu(self.tcn1(self.gcn1(x, label=label, mix_top=mix_top)) + self.residual(x))
            return y


def get_attn_map_s(x, e_lambda=1e-4):
        NM, C, T, V = x.size()
        num = V * T - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / num + e_lambda)) + 0.5
        att_map = torch.sigmoid(y)
        att_map_s = att_map.mean(dim=[1, 2])
        # N * M, V
        return att_map_s