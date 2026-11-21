#Author: Yiming Yang
#Date: 2026-11
#License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from mamba_ssm import Mamba
torch.backends.mha.set_fastpath_enabled(False)

class BiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2
    ):
        super().__init__()

        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )


        self.proj = nn.Linear(
            d_model * 2,
            d_model
        )


    def forward(self,x):

        x = x.contiguous()

        out_f = self.mamba_forward(x).contiguous()


        x_reverse = torch.flip(
            x,
            dims=[1]
        ).contiguous()


        out_b = self.mamba_backward(
            x_reverse
        ).contiguous()


        out_b = torch.flip(
            out_b,
            dims=[1]
        ).contiguous()


        out = torch.cat(
            [
                out_f,
                out_b
            ],
            dim=-1
        ).contiguous()


        out = self.proj(out)

        return out.contiguous()

class SingleBlock_BLSTM(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, nhead, dropout=0, bidirectional=True, eps=1e-8):
        super(SingleBlock_BLSTM, self).__init__()

        self.feature_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_direction = int(bidirectional) + 1

        self.att_freq = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_freq = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_freq = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)
        self.fc_freq = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_freq = nn.LayerNorm(self.feature_dim, eps=eps)

        self.att_time = nn.MultiheadAttention(self.feature_dim, self.nhead, batch_first=True)
        self.norm_att_time = nn.LayerNorm(self.feature_dim, eps=eps)
        self.rnn_time = getattr(nn, rnn_type)(self.feature_dim, self.hidden_dim, 1, dropout=dropout,
                                              batch_first=True, bidirectional=bidirectional)
        self.fc_time = nn.Linear(self.hidden_dim * self.num_direction, self.feature_dim)
        self.norm_time = nn.LayerNorm(self.feature_dim, eps=eps)

    def forward(self, input):
        # input: (Batch, Channel, Freq, Time)

        batch_size, channel, freq, time = input.size()

        input_freq = input.permute(0, 3, 2, 1).contiguous().view(batch_size * time, freq, -1)
        att_out_freq, _ = self.att_freq(input_freq, input_freq, input_freq)
        att_out_freq = att_out_freq + input_freq
        att_out_freq = self.norm_att_freq(att_out_freq)
        rnn_out_freq, _ = self.rnn_freq(att_out_freq)
        fc_out_freq = self.fc_freq(rnn_out_freq)
        fc_out_freq = fc_out_freq + att_out_freq
        fc_out_freq = self.norm_freq(fc_out_freq)
        fc_out_freq = fc_out_freq.contiguous().view(batch_size, time, freq, -1).permute(0, 3, 2, 1)

        input_time = fc_out_freq.permute(0, 2, 3, 1).contiguous().view(batch_size * freq, time, -1)
        att_out_time, _ = self.att_time(input_time, input_time, input_time)
        att_out_time = att_out_time + input_time
        att_out_time = self.norm_att_time(att_out_time)
        rnn_out_time, _ = self.rnn_time(att_out_time)
        fc_out_time = self.fc_time(rnn_out_time)
        fc_out_time = fc_out_time + att_out_time
        fc_out_time = self.norm_time(fc_out_time)
        fc_out_time = fc_out_time.contiguous().view(batch_size, freq, time, -1).permute(0, 3, 1, 2)

        return fc_out_time


class SingleBlock_Mamba(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            nhead,
            dropout=0,
            eps=1e-8):
        super(SingleBlock_Mamba, self).__init__()

        self.feature_dim = input_dim

        # ======================
        # Frequency branch
        # ======================

        self.att_freq = nn.MultiheadAttention(
            input_dim,
            nhead,
            batch_first=True
        )

        self.norm_att_freq = nn.LayerNorm(
            input_dim,
            eps=eps
        )

        # BLSTM replaced by Mamba

        self.mamba_freq = Mamba(
            d_model=input_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.fc_freq = nn.Linear(
            input_dim,
            input_dim
        )

        self.norm_freq = nn.LayerNorm(
            input_dim,
            eps=eps
        )

        # ======================
        # Time branch
        # ======================

        self.att_time = nn.MultiheadAttention(
            input_dim,
            nhead,
            batch_first=True
        )

        self.norm_att_time = nn.LayerNorm(
            input_dim,
            eps=eps
        )

        # BLSTM replaced by Mamba

        self.mamba_time = BiMamba(
            d_model=input_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.fc_time = nn.Linear(
            input_dim,
            input_dim
        )

        self.norm_time = nn.LayerNorm(
            input_dim,
            eps=eps
        )

    def forward(self, input):
        # input:
        # B,C,F,T

        batch_size, channel, freq, time = input.size()

        #################################
        # Frequency modeling
        #################################

        input_freq = (
            input
            .permute(0, 3, 2, 1)
            .contiguous()
            .view(batch_size * time,
                  freq,
                  channel)
        )

        # MHA (unchanged)

        att_out_freq, _ = self.att_freq(
            input_freq,
            input_freq,
            input_freq
        )

        att_out_freq = att_out_freq + input_freq

        att_out_freq = self.norm_att_freq(
            att_out_freq
        )

        # BLSTM -> Mamba

        mamba_out_freq = self.mamba_freq(
            att_out_freq
        )

        fc_out_freq = self.fc_freq(
            mamba_out_freq
        )

        fc_out_freq = fc_out_freq + att_out_freq

        fc_out_freq = self.norm_freq(
            fc_out_freq
        )

        fc_out_freq = (
            fc_out_freq
            .contiguous()
            .view(batch_size,
                  time,
                  freq,
                  channel)
            .permute(0, 3, 2, 1)
        )

        #################################
        # Time modeling
        #################################

        input_time = (
            fc_out_freq
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size * freq,
                  time,
                  channel)
        )

        # MHA unchanged

        att_out_time, _ = self.att_time(
            input_time,
            input_time,
            input_time
        )

        att_out_time = att_out_time + input_time

        att_out_time = self.norm_att_time(
            att_out_time
        )

        # BLSTM -> Mamba

        mamba_out_time = self.mamba_time(
            att_out_time
        )

        fc_out_time = self.fc_time(
            mamba_out_time
        )

        fc_out_time = fc_out_time + att_out_time

        fc_out_time = self.norm_time(
            fc_out_time
        )

        output = (
            fc_out_time
            .contiguous()
            .view(batch_size,
                  freq,
                  time,
                  channel)
            .permute(0, 3, 1, 2)
        )

        return output


class Separator(nn.Module):

    def __init__(self, input_dim, output_dim,
                 bottleneck_dim,
                 hidden_dim,
                 nhead,
                 num_layer,
                 eps=1e-8):

        super(Separator,self).__init__()

        self.LayerNormalization = nn.GroupNorm(
            1,
            input_dim,
            eps=eps
        )

        self.bottleneck = nn.Conv2d(
            input_dim,
            bottleneck_dim,
            1
        )
        self.Stack = nn.ModuleList()

        for i in range(num_layer):

            if i % 2 == 0:
                # 第 0, 2, 4 ... 层用 BLSTM
                self.Stack.append(
                    SingleBlock_BLSTM(
                        "LSTM",
                        bottleneck_dim,
                        hidden_dim,
                        nhead
                    )
                )
            else:
                # 第 1, 3, 5 ... 层用 Mamba
                self.Stack.append(
                    SingleBlock_Mamba(
                        bottleneck_dim,
                        hidden_dim,
                        nhead
                    )
                )
        self.output_con2d = nn.Conv2d(
            bottleneck_dim,
            output_dim,
            1
        )


    def forward(self,input):
        output = self.bottleneck(
            self.LayerNormalization(input)
        )
        for block in self.Stack:
            output = block(output)

        output = self.output_con2d(output)


        return output


class FreqNet(nn.Module):
    def __init__(self, enc_dim=256, bottleneck_dim=64, hidden_dim=128, nhead=4, num_layer=6,
                 sr=8000, win_size=256, fft_len=256, win_type='hann'):
        super(FreqNet, self).__init__()

        self.enc_dim = enc_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.nhead = nhead
        self.sr = sr
        self.win_size = win_size
        self.hop_size = win_size // 2

        self.fft_len = fft_len
        self.win_type = win_type

        self.softmax = nn.Softmax(dim=-2)
        self.alpha = nn.Parameter(torch.tensor(0., dtype=torch.float32))

        self.encoder = nn.Conv2d(4, self.enc_dim, kernel_size=7, padding=3)
        self.encoder_nonlinearity = nn.ReLU()
        self.separator = Separator(self.enc_dim, self.enc_dim, self.bottleneck_dim, self.hidden_dim, self.nhead, self.num_layer)
        self.nonlinearity = nn.ReLU()
        self.decoder_refine = nn.Sequential(
            # Local TF modeling
            nn.Conv2d(
                self.enc_dim,
                self.enc_dim,
                kernel_size=3,
                padding=1,
                groups=self.enc_dim
            ),

            nn.GELU(),

            nn.Conv2d(
                self.enc_dim,
                self.enc_dim,
                kernel_size=1
            )
        )
        # Final complex spectrum projection
        self.decoder = nn.Conv2d(
            self.enc_dim,
            2,
            kernel_size=1
        )

    def pad_signal(self, input):
        #input size: (Batch, Time) or (Batch, 1, Time)

        if input.dim() not in [2, 3]:
            raise RuntimeError('Input can only be 2 or 3 dimensional.')

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win_size - (self.hop_size + nsample % self.win_size) % self.win_size
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_extra = torch.zeros(batch_size, 1, self.hop_size).type(input.type())
        input = torch.cat([pad_extra, input, pad_extra], 2)
        return input, rest

    def init_kernels(self, inverse=False):
        if self.win_type is None:
            window = np.ones(self.win_len)
        else:
            window = get_window(self.win_type, self.win_size, fftbins=True)

        fourier_basis = np.fft.rfft(np.eye(self.fft_len))[: self.win_size]
        real_kernel = np.real(fourier_basis)
        imag_kernel = np.imag(fourier_basis)
        kernel = np.concatenate([real_kernel, imag_kernel], 1).T

        if inverse:
            kernel = np.linalg.pinv(kernel).T

        kernel = kernel * window
        kernel = kernel[:, None, :]
        return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))

    def ConvSTFT(self, input):
        kernel, _ = self.init_kernels()
        output = F.conv1d(input, kernel.to(input.device), stride=self.hop_size)

        return output

    def ConviSTFT(self, input):
        kernel, window = self.init_kernels(inverse=True)
        output = F.conv_transpose1d(input, kernel.to(input.device), stride=self.hop_size)

        t = window.repeat(1, 1, input.size(-1)) ** 2
        enframe = torch.eye(self.win_size).unsqueeze(1)
        coff = F.conv_transpose1d(t, enframe, stride=self.hop_size)
        output = output / (coff.to(input.device) + 1e-8)

        return output

    def FeaCompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = torch.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = torch.abs(complex_spectrum).unsqueeze(1) ** factor
        phase = torch.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        output = torch.cat((real, imag), dim=1)

        return output

    def FeaDecompression(self, input, factor=0.5):
        input_change = input.float()
        complex_spectrum = torch.complex(input_change[:, 0, :, :], input_change[:, 1, :, :])
        magnitude = torch.abs(complex_spectrum).unsqueeze(1) ** (1 / factor)
        phase = torch.angle(complex_spectrum).unsqueeze(1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        output = torch.cat((real, imag), dim=1)

        return output

    def ComputeSimilarity(self, input, enrollment):
        att = enrollment.transpose(-2, -1) @ input
        att = self.softmax(att)
        output = enrollment @ att

        return output.unsqueeze(0).unsqueeze(0)

    def forward(self, input, enrollment):
        if enrollment.dim() == 2:
            enrollment = enrollment.unsqueeze(1)
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)

        output= self.ConvSTFT(output).contiguous().view(batch_size, 2, self.fft_len // 2 + 1, -1)
        output = self.FeaCompression(output)

        similarity = []
        for i in range(batch_size):
            temp, _ = self.pad_signal(enrollment[i])
            temp = self.ConvSTFT(temp).contiguous().view(
                1, 2, self.fft_len // 2 + 1, -1
            )
            temp = self.FeaCompression(temp)

            # similarity: [1, 2, F, T]
            sim = torch.cat([
                self.ComputeSimilarity(
                    output[i, 0, ...],
                    temp[0, 0, ...]
                ),
                self.ComputeSimilarity(
                    output[i, 1, ...],
                    temp[0, 1, ...]
                )
            ], dim=1)

            # G: [1, 2, F, 1]
            G = temp.mean(dim=-1, keepdim=True)

            # G modulates similarity
            sim = sim * (1 + self.alpha * G)

            similarity.append(sim)

        similarity = torch.cat(similarity, dim=0)

        enc_output = self.encoder_nonlinearity(self.encoder(torch.cat((output, similarity), dim=1)))
        masks = self.nonlinearity(self.separator(enc_output).view(batch_size, 1, self.enc_dim, self.fft_len // 2 + 1, -1))
        masked_output = enc_output.unsqueeze(1) * masks

        masked_output = masked_output.view(
            batch_size,
            self.enc_dim,
            self.fft_len // 2 + 1,
            -1
        )
        # ==========================================
        # Local Time-Frequency Refinement
        # ==========================================
        refined = self.decoder_refine(masked_output)

        masked_output = masked_output + refined

        # Final decoder
        dec_output = self.decoder(masked_output)

        output = self.FeaDecompression(dec_output)
        output = self.ConviSTFT(output.contiguous().view(batch_size, self.fft_len + 2, -1))
        output = output[:, :, self.hop_size:-(rest + self.hop_size)].contiguous()
        output = output.view(batch_size, 1, -1)
        output = output.squeeze(1)
        return output

if __name__ == "__main__":
    import torch
    from thop import profile, clever_format

    device = torch.device("cuda:0")
    conf = {
        "sr": 16000,
        "win_size": 512,
        "fft_len": 512,
        "enc_dim": 256,
        "bottleneck_dim": 64,
        "hidden_dim": 128,
        "nhead": 4,
        "num_layer": 6,
        "win_type": 'hann'
    }

    model = FreqNet(**conf).to(device)

    mix = torch.randn(
        1, 16000,
        device=device
    )
    aux = torch.randn(
        1, 16000,
        device=device
    )
    model.eval()
    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(mix, aux),
            verbose=False
        )
    macs, params = clever_format(
        [macs, params],
        "%.3f"
    )
    print(macs, params)
