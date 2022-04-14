import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from thop import clever_format
from thop import profile
from utils import capture_init


EPS = 1e-8


def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class STFT(nn.Module):
    def __init__(self, fftsize=256, window_size=88, stride=44, win_type="default", trainable=False, online=False):
        super(STFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride

        if win_type=="default": # sin window
            self.window_func = np.sqrt(np.hanning(self.window_size))
        elif win_type=="hanning":
            self.window_func = np.hanning(self.window_size)


        fcoef_r = np.zeros((self.fftsize//2 + 1, 2, self.window_size))
        fcoef_i = np.zeros((self.fftsize//2 + 1, 2, self.window_size))

        for w in range(self.fftsize//2+1):
            for t in range(self.window_size):
                fcoef_r[w, 0, t] = np.cos(2. * np.pi * w * t / self.fftsize)
                fcoef_i[w, 0, t] = -np.sin(2. * np.pi * w * t / self.fftsize)


        fcoef_r = fcoef_r * self.window_func
        fcoef_i = fcoef_i * self.window_func

        self.fcoef_r = torch.tensor(fcoef_r, dtype=torch.float)
        self.fcoef_i = torch.tensor(fcoef_i, dtype=torch.float)

        self.encoder_r = nn.Conv1d(2, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_i = nn.Conv1d(2, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)


        self.encoder_r.weight = torch.nn.Parameter(self.fcoef_r)
        self.encoder_i.weight = torch.nn.Parameter(self.fcoef_i)

        if trainable:
            self.encoder_r.weight.requires_grad = True
            self.encoder_i.weight.requires_grad = True
        else:
            self.encoder_r.weight.requires_grad = False
            self.encoder_i.weight.requires_grad = False


        # for online
        if online:
            #self.input_buffer = th.tensor(np.zeros([1,1,self.window_size]),dtype=th.float,device=th.device('cpu'))
            self.input_buffer = torch.tensor(np.zeros([1, 1, self.window_size]),dtype=torch.float)


    def set_buffer_device(self, device):
        self.input_buffer = self.input_buffer.to(device)


        return


    def forward(self, input):


        spec_r = self.encoder_r(input)
        spec_i = self.encoder_i(input)

        x_spec_real = spec_r[:,1:,:] # remove DC
        x_spec_imag = spec_i[:,1:,:] # remove DC
        output = torch.cat([x_spec_real,x_spec_imag],dim=1)

        return output



class CDEHTCN(nn.Module):
    @capture_init
    def __init__(self,
                 sources,
                 N=256,
                 L=88,
                 B=256,
                 H=512,
                 P=3,
                 X=8,
                 R=4,
                 audio_channels=2,
                 norm_type="SN",
                 causal=False,
                 mask_nonlinear='relu',
                 samplerate=44100,
                 segment_length=44100 * 2 * 4):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: SN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(CDEHTCN, self).__init__()
        self.sources = sources
        self.C = len(sources)
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R = N, L, B, H, P, X, R
        self.stage_decoder = nn.ModuleList()
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.segment_length = segment_length
        # Components
        self.CDEncoder = CDEncoder(N, L, audio_channels)
        self.separator = HTCN(N, B, H, P, X, R, self.C, norm_type, causal, mask_nonlinear)
        for i in range(self.R):
            self.stage_decoder.append(Decoder(N, L, audio_channels))

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture):

        mixture_w = self.CDEncoder(mixture)
        est_stage_mask = self.separator(mixture_w)
        est_list = []
        for r in range(self.R):
            est_source = self.stage_decoder[r](mixture_w, est_stage_mask[r])
            T_origin = mixture.size(-1)
            T_conv = est_source.size(-1)
            est_source = F.pad(est_source, (0, T_origin - T_conv))
            est_list.append(est_source)

        return est_list


class CDEncoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super(CDEncoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        # Components
        self.conv1d = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)
        self.stft = STFT(self.N, self.L, self.L//2, trainable=False, online=False)

        self.linear1 = nn.Conv1d(N, N, kernel_size=1, bias=False)
        self.linear2 = nn.Conv1d(N, N, kernel_size=1, bias=False)
        self.linear3 = nn.Conv1d(2*N, N, kernel_size=1, bias=False)


    def forward(self, x):
        stft_feature = self.linear1(self.stft(x))
        conv_feature = self.linear2(self.conv1d(x))
        fusion_feature = self.linear3(torch.cat([stft_feature, conv_feature], dim=1))
        ratio_mask1 = torch.sigmoid(fusion_feature)
        ratio_mask2 = 1 - ratio_mask1

        conv_out = conv_feature * ratio_mask1
        stft_out = stft_feature * ratio_mask2

        fusion_out = conv_out + stft_out
        out = F.relu(stft_feature + conv_feature + fusion_out)

        return out


class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        # Components
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, ac * L]
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x ac x T
        return est_source


class HTCN(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="SN", causal=False, mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: SN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(HTCN, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear

        self.layer_norm = ChannelwiseLayerNorm(N)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        self.multi_stage_TCN = nn.ModuleList([])
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(B,
                                  H,
                                  P,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  causal=causal)
                ]
            self.multi_stage_TCN.append(nn.ModuleList([*blocks]))

        self.fuse = nn.ModuleList()
        for r in range(R):
            self.fuse.append(nn.Sequential(nn.Conv1d(B*(X+1), B, 1, bias=False),
                                           nn.PReLU(),
                                           chose_norm(norm_type, B)
            ))

        self.SE_block = nn.ModuleList()
        for r in range(R):
            modules = [SqueezeAndExcite(B, B)]
            self.SE_block.append(nn.Sequential(*modules))

        self.out_conv1x1 = nn.ModuleList()
        for r in range(R):
            self.out_conv1x1.append(nn.Conv1d(B, C * N, 1, bias=False))



    def forward(self, mixture_w):

        M, N, K = mixture_w.size()
        score = self.layer_norm(mixture_w)
        score = self.bottleneck_conv1x1(score)
        stage_mask = []
        residual = score
        for i, layer in enumerate(self.multi_stage_TCN):
            skip = [score]
            if not i == 0:
                score = score + residual
            for j, sublayer in enumerate(layer):
                score = sublayer(score)
                skip.append(score)
            score = self.fuse[i](torch.cat(skip, dim=1))
            score = self.SE_block[i](score)
            mask = self.out_conv1x1[i](score)
            mask = mask.view(M, self.C, N, K) 
            if self.mask_nonlinear == 'softmax':
                est_mask = F.softmax(mask, dim=1)
            elif self.mask_nonlinear == 'relu':
                est_mask = F.relu(mask)
            else:
                raise ValueError("Unsupported mask non-linear function")
            stage_mask.append(est_mask) 

        return stage_mask


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x*self.relu6(x+3)/6


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, t = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1)
        return out * x



class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="SN",
                 causal=False):
        super(TemporalBlock, self).__init__()
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding, dilation, norm_type, causal)
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        residual = x

        out = self.net(x)
        return out + residual 

class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="SN",
                 causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, B, K] -> [M, B, K]
        # se_module = SqueezeAndExcite(in_channels, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
   
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    if norm_type == "SN":
        return SwitchNorm2d(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "gLN":
        return return GlobalLayerNorm(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        return x * self.weight + self.bias



def test_model(model):
    x = torch.rand(2, 2, 176400)  # (batch, length)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:{}    params:{}'.format(flops, params))
    print(model)
    y = model(x)
    print(y[0].shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model = CDEHTCN(sources=["drums", "bass", "other", "vocals"])
    test_model(model)
